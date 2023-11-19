#include <my_simulator/RosSim/MagnetoRosNode.hpp>


MagnetoRosNode::MagnetoRosNode(ros::NodeHandle& nh, const dart::simulation::WorldPtr& _world)
    : nh_(nh), dart::gui::osg::WorldNode(_world), count_(0), t_(0.0), servo_rate_(0) {
    world_ = _world;

    // ---- GET SKELETON
    robot_ = world_->getSkeleton("magneto");
    ground_ = world_->getSkeleton("ground_skeleton");
    std::cout<< "Magneto mass : " <<  robot_->getMass() << std::endl;
    
    // ---- GET INFO FROM SKELETON
    // CheckRobotSkeleton(robot_);
    n_dof_ = robot_->getNumDofs();        

    // ---- PLOT?
    b_plot_result_ = true;

    // ---- SET POSITION LIMIT
    // setPositionLimitEnforced

    // contact dinstance
    contact_threshold_ = 0.005;
    contact_distance_[MagnetoBodyNode::AL_foot_link] = 0.05;
    contact_distance_[MagnetoBodyNode::BL_foot_link] = 0.05;
    contact_distance_[MagnetoBodyNode::AR_foot_link] = 0.05;
    contact_distance_[MagnetoBodyNode::BR_foot_link] = 0.05;

    trq_cmd_ = Eigen::VectorXd::Zero(n_dof_);

    // ---- SET INTERFACE
    interface_ = new MagnetoInterface();
    sensor_data_ = new MagnetoSensorData();
    command_ = new MagnetoCommand();

    sensor_data_->R_ground = ground_->getBodyNode("ground_link")
                                    ->getWorldTransform().linear();

    // ---& RL PARAMS
    if (!nh_.param<bool>("/magneto/rl/active", should_use_rl_, false));
    if (should_use_rl_) {
        if (!nh_.param<std::string>("/magneto/simulation/walkset", walkset_, "config/Magneto/USERCONTROLWALK.yaml"));
        nh_.param<double>("/magneto/simulation/resume_duration", resume_duration_, 3.0);
    } else {
        nh_.param<std::string>("/magneto/simulation/walkset", walkset_, "config/Magneto/SIMULATIONWALK.yaml");
    }
    num_rl_commands_received_ = 0;
    status_pub_ = nh_.advertise<std_msgs::String>("/magneto/node/status", 1, this);
    
    if (should_use_rl_) {
        next_step_client_ = nh_.serviceClient<magneto_rl::FootPlacement>("/determine_next_step");
        // trigger_rl_server_ = nh_.advertiseService("trigger_rl_update", &MagnetoRosNode::enterRLPlugin, this);
        trigger_rl_server_ = nh_.advertiseService("trigger_rl_update", &MagnetoRosNode::enterRLPlugin, this);
        report_state_server_ = nh_.advertiseService("get_magneto_state", &MagnetoRosNode::reportStateInformation, this);
        action_command_server_ = nh_.advertiseService("set_magneto_action", &MagnetoRosNode::updateActionCommand, this);
        sim_reset_server_ = nh_.advertiseService("reset_magneto_sim", &MagnetoRosNode::resetSim, this);
        magnetism_update_client_ = nh_.serviceClient<magneto_rl::MagnetismModifierRequest>("/get_magnetism_modifier");
    }

    // ---- SET control parameters %% motion script
    run_mode_ = ((MagnetoInterface*)interface_)->getRunMode();
    SetParams_();
    ReadMotions_();


    // ---- SET TORQUE LIMIT
    trq_lb_ = Eigen::VectorXd::Constant(n_dof_, -torque_limit_);
    trq_ub_ = Eigen::VectorXd::Constant(n_dof_, torque_limit_);
    // trq_lb_ = robot_->getForceLowerLimits();
    // trq_ub_ = robot_->getForceUpperLimits();
    // std::cout<<"trq_lb_ = "<<trq_lb_.transpose() << std::endl;
    // std::cout<<"trq_ub_ = "<<trq_ub_.transpose() << std::endl;  

    plugin_updated_ = true;
    ready_for_rl_input_ = false;
    sim_resume_time_ = ros::Time::now().toSec();
}

MagnetoRosNode::~MagnetoRosNode() {
    delete interface_;
    delete sensor_data_;
    delete command_;
}


void MagnetoRosNode::CheckRobotSkeleton(const dart::dynamics::SkeletonPtr& skel){
    size_t n_bodynode = skel->getNumBodyNodes();
    std::string bn_name;

    // check inertia
    Eigen::MatrixXd I_tmp;
    bool b_check;
    for(size_t idx = 0; idx<n_bodynode; ++idx){
        bn_name = skel->getBodyNode(idx)->getName();
        // I_tmp = skel->getBodyNode(idx)->getInertia().getMoment();
        // my_utils::pretty_print(I_tmp, std::cout, bn_name);
        // b_check=skel->getBodyNode(idx)->getInertia().verify(true, 1e-5);
        I_tmp = skel->getBodyNode(idx)->getArticulatedInertia();
        my_utils::pretty_print(I_tmp, std::cout, bn_name);
    }    
}

void MagnetoRosNode::customPostStep() {
}

void MagnetoRosNode::enableButtonFlag(uint16_t key) {
    std::cout << "button(" << (char)key << ") pressed handled @ MagnetoRosNode::enableButtonFlag" << std::endl;
    ((MagnetoInterface*)interface_) -> interrupt_ -> setFlags(key);
}

// & Input for changing global variable to enter the pluginInterface
// bool MagnetoInterface::enterRLPlugin (magneto_rl::FootPlacementAction::Request &req, magneto_rl::FootPlacementAction::Response &res) {
bool MagnetoRosNode::enterRLPlugin (std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res) {
    // ROS_INFO_STREAM("Setting ready_for_rl_input to true!");
    ready_for_rl_input_ = true;
    res.success=true;
    return true;
}

// & Primary RL plugin interface function
void MagnetoRosNode::pluginInterface() {
    plugin_updated_ = true;
    // ready_for_rl_input_ = false;
    std::cout<<" plugin_updated_  = " << plugin_updated_ << std::endl;
    ROS_WARN_STREAM("Inside pluginInterface()");
    std_msgs::String status;
    status.data = "In plot foot step result";
    status_pub_.publish(status);

    // & filling in state information
    magneto_rl::FootPlacement srv;
    // Eigen::Quaternion<double> quat_ground 
    //                         = Eigen::Quaternion<double>( 
    //                             ground_->getBodyNode("ground_link")
    //                                     ->getWorldTransform().linear());
    Eigen::Quaternion<double> q_base 
                            = Eigen::Quaternion<double>( 
                                robot_->getBodyNode(MagnetoBodyNode::base_link)
                                        ->getWorldTransform().linear());
    Eigen::VectorXd p_base = robot_->getBodyNode(MagnetoBodyNode::base_link)->getWorldTransform().translation();
    Eigen::VectorXd p_ar = robot_->getBodyNode(MagnetoBodyNode::AR_foot_link)->getWorldTransform().translation();
    Eigen::VectorXd p_al = robot_->getBodyNode(MagnetoBodyNode::AL_foot_link)->getWorldTransform().translation();
    Eigen::VectorXd p_bl = robot_->getBodyNode(MagnetoBodyNode::BL_foot_link)->getWorldTransform().translation();
    Eigen::VectorXd p_br = robot_->getBodyNode(MagnetoBodyNode::BR_foot_link)->getWorldTransform().translation();

    geometry_msgs::Pose base_pose;
    base_pose.position.x = p_base.x();
    base_pose.position.y = p_base.y();
    base_pose.position.z = p_base.z();
    base_pose.orientation.w = q_base.w();
    base_pose.orientation.x = q_base.x();
    base_pose.orientation.y = q_base.y();
    base_pose.orientation.z = q_base.z();

    geometry_msgs::Point point_ar;
    point_ar.x = p_ar.x();
    point_ar.y = p_ar.y();
    point_ar.z = p_ar.z();

    geometry_msgs::Point point_al;
    point_al.x = p_al.x();
    point_al.y = p_al.y();
    point_al.z = p_al.z();

    geometry_msgs::Point point_bl;
    point_bl.x = p_bl.x();
    point_bl.y = p_bl.y();
    point_bl.z = p_bl.z();

    geometry_msgs::Point point_br;
    point_br.x = p_br.x();
    point_br.y = p_br.y();
    point_br.z = p_br.z();

    srv.request.base_pose = base_pose;
    srv.request.p_ar = point_ar;
    srv.request.p_al = point_al;
    srv.request.p_bl = point_bl;
    srv.request.p_br = point_br;

    if (!next_step_client_.call(srv)) {
        ROS_ERROR_STREAM("Failed to call determine_next_step service");
    }

    int link_idx;
    MOTION_DATA md_temp;

    Eigen::VectorXd pos_temp(3);
    Eigen::VectorXd ori_temp(4);
    ori_temp << 1, 0, 0, 0;
    pos_temp << srv.response.pos.x, srv.response.pos.y, srv.response.pos.z;
    md_temp.motion_period = 0.4;
    md_temp.swing_height = 0.05;

    link_idx = srv.response.link_idx;

    bool is_bodyframe {true};
    md_temp.pose = POSE_DATA(pos_temp, ori_temp, is_bodyframe);

    ((MagnetoInterface*)interface_)->ClearWalkMotions();
    ((MagnetoInterface*)interface_)->AddAndExecuteMotion(link_idx, md_temp);

    num_rl_commands_received_++;
    ready_for_rl_input_ = false;
}

// TODO fill in these functions and make sure they work as intended
bool MagnetoRosNode::reportStateInformation(magneto_rl::ReportMagnetoState::Request &req, magneto_rl::ReportMagnetoState::Response &res) {
    // WIP
    // . Ground Pose
    Eigen::Quaternion<double> q_ground
                            = Eigen::Quaternion<double>( 
                                ground_->getBodyNode("ground_link")
                                        ->getWorldTransform().linear() );
    Eigen::VectorXd p_ground = ground_->getBodyNode("ground_link")->getWorldTransform().translation();
    
    geometry_msgs::Pose ground_pose;
    ground_pose.position.x = p_ground.x();
    ground_pose.position.y = p_ground.y();
    ground_pose.position.z = p_ground.z();
    ground_pose.orientation.w = q_ground.w();
    ground_pose.orientation.x = q_ground.x();
    ground_pose.orientation.y = q_ground.y();
    ground_pose.orientation.z = q_ground.z();

    // . Base pose
    Eigen::Quaternion<double> q_base 
                            = Eigen::Quaternion<double>( 
                                robot_->getBodyNode(MagnetoBodyNode::base_link)
                                        ->getWorldTransform().linear());
    Eigen::VectorXd p_base = robot_->getBodyNode(MagnetoBodyNode::base_link)->getWorldTransform().translation();
    
    geometry_msgs::Pose base_pose;
    base_pose.position.x = p_base.x();
    base_pose.position.y = p_base.y();
    base_pose.position.z = p_base.z();
    base_pose.orientation.w = q_base.w();
    base_pose.orientation.x = q_base.x();
    base_pose.orientation.y = q_base.y();
    base_pose.orientation.z = q_base.z();

    // . ar foot information
    magneto_rl::FootState state_ar;
    Eigen::Quaternion<double> q_ar 
                            = Eigen::Quaternion<double>( 
                                robot_->getBodyNode(MagnetoBodyNode::AR_foot_link)
                                        ->getWorldTransform().linear());
    Eigen::VectorXd p_ar = robot_->getBodyNode(MagnetoBodyNode::AR_foot_link)->getWorldTransform().translation();
    
    geometry_msgs::Pose pose_ar;
    pose_ar.position.x = p_ar.x();
    pose_ar.position.y = p_ar.y();
    pose_ar.position.z = p_ar.z();
    pose_ar.orientation.w = q_ar.w();
    pose_ar.orientation.x = q_ar.x();
    pose_ar.orientation.y = q_ar.y();
    pose_ar.orientation.z = q_ar.z();

    state_ar.pose = pose_ar;
    
    // . al foot information
    magneto_rl::FootState state_al;
    Eigen::Quaternion<double> q_al 
                            = Eigen::Quaternion<double>( 
                                robot_->getBodyNode(MagnetoBodyNode::AL_foot_link)
                                        ->getWorldTransform().linear());
    Eigen::VectorXd p_al = robot_->getBodyNode(MagnetoBodyNode::AL_foot_link)->getWorldTransform().translation();
    
    geometry_msgs::Pose pose_al;
    pose_al.position.x = p_al.x();
    pose_al.position.y = p_al.y();
    pose_al.position.z = p_al.z();
    pose_al.orientation.w = q_al.w();
    pose_al.orientation.x = q_al.x();
    pose_al.orientation.y = q_al.y();
    pose_al.orientation.z = q_al.z();

    state_al.pose = pose_al;

    // . bl foot information
    magneto_rl::FootState state_bl;
    Eigen::Quaternion<double> q_bl
                            = Eigen::Quaternion<double>( 
                                robot_->getBodyNode(MagnetoBodyNode::BL_foot_link)
                                        ->getWorldTransform().linear());
    Eigen::VectorXd p_bl = robot_->getBodyNode(MagnetoBodyNode::BL_foot_link)->getWorldTransform().translation();
    
    geometry_msgs::Pose pose_bl;
    pose_bl.position.x = p_bl.x();
    pose_bl.position.y = p_bl.y();
    pose_bl.position.z = p_bl.z();
    pose_bl.orientation.w = q_bl.w();
    pose_bl.orientation.x = q_bl.x();
    pose_bl.orientation.y = q_bl.y();
    pose_bl.orientation.z = q_bl.z();

    state_bl.pose = pose_bl;

    // . br foot information
    magneto_rl::FootState state_br;
    Eigen::Quaternion<double> q_br
                            = Eigen::Quaternion<double>( 
                                robot_->getBodyNode(MagnetoBodyNode::BR_foot_link)
                                        ->getWorldTransform().linear());
    Eigen::VectorXd p_br = robot_->getBodyNode(MagnetoBodyNode::BR_foot_link)->getWorldTransform().translation();

    geometry_msgs::Pose pose_br;
    pose_br.position.x = p_br.x();
    pose_br.position.y = p_br.y();
    pose_br.position.z = p_br.z();
    pose_br.orientation.w = q_br.w();
    pose_br.orientation.x = q_br.x();
    pose_br.orientation.y = q_br.y();
    pose_br.orientation.z = q_br.z();

    state_br.pose = pose_br;

    // . composition into service response
    res.ground_pose = ground_pose;
    res.body_pose = base_pose;
    res.AR_state = state_ar;
    res.AL_state = state_al;
    res.BL_state = state_bl;
    res.BR_state = state_br;

    return 1;
}

bool MagnetoRosNode::updateActionCommand(magneto_rl::UpdateMagnetoAction::Request &req, magneto_rl::UpdateMagnetoAction::Response &res) {
    // WIP
    ready_for_rl_input_ = true;

    latest_link_idx_ = req.link_idx;
    Eigen::VectorXd pos_command(3);
    Eigen::VectorXd ori_command(4);
    ori_command << req.foot_pose.orientation.w, req.foot_pose.orientation.x, req.foot_pose.orientation.y, req.foot_pose.orientation.z;
    pos_command << req.foot_pose.position.x, req.foot_pose.position.y, req.foot_pose.position.z;
    latest_motion_data_.motion_period = 0.4;
    latest_motion_data_.swing_height = 0.05;
    bool is_bodyframe {true};
    latest_motion_data_.pose = POSE_DATA(pos_command, ori_command, is_bodyframe);

    Eigen::VectorXd p_base = robot_->getBodyNode(MagnetoBodyNode::base_link)->getWorldTransform().translation();
    magneto_rl::MagnetismModifierRequest srv;
    srv.request.point.x = p_base[0];
    srv.request.point.y = p_base[1];
    if (magnetism_update_client_.call(srv)) {
        magnetism_modifier_ =  srv.response.modifier;
    }

    res.success = true;
    return 1;
}

void MagnetoRosNode::enactActionCommand () {
    plugin_updated_ = true;
    ((MagnetoInterface*)interface_)->ClearWalkMotions();
    ((MagnetoInterface*)interface_)->AddAndExecuteMotion(latest_link_idx_, latest_motion_data_);
    num_rl_commands_received_++;
    ready_for_rl_input_ = false;
}

bool MagnetoRosNode::resetSim(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res) {
    // TODO
    // while ((ros::Time::now().toSec() - sim_resume_time_) <= 10) {
    //     ROS_INFO_STREAM("WAITING!");
    // }

    Eigen::VectorXd starting_pose;
    starting_pose << 0.0, 0.0, 0.15, 0.0, 1.0, 0.0;
    Eigen::VectorXd q = robot_->getPositions();
    q.segment(0,6) = starting_pose.head(6);

    std::string coxa("coxa_joint");
    std::string femur("femur_joint");
    std::string tibia("tibia_joint");
    std::string foot1("foot_joint_1");
    std::string foot2("foot_joint_2");
    std::string foot3("foot_joint_3");

    const std::array<std::string, 4> foot_name = {"AL_", "BL_", "AR_", "BR_"};

    double femur_joint_init = 1./10.*M_PI_2; // -1./10.*M_PI_2;
    double tibia_joint_init = -11./10.*M_PI_2; // -9./10.*M_PI_2;

    for(int i=0; i<foot_name.size(); i++) {
        q[robot_->getDof(foot_name[i] + coxa)->getIndexInSkeleton()] = 0.0;
        q[robot_->getDof(foot_name[i] + femur)->getIndexInSkeleton()] = femur_joint_init;
        q[robot_->getDof(foot_name[i] + tibia)->getIndexInSkeleton()] = tibia_joint_init;
        q[robot_->getDof(foot_name[i] + foot1)->getIndexInSkeleton()] = 0.0;
        q[robot_->getDof(foot_name[i] + foot2)->getIndexInSkeleton()] = 0.0;
        q[robot_->getDof(foot_name[i] + foot3)->getIndexInSkeleton()] = 0.0;
    }

    robot_->setPositions(q);

    res.success = false;
    return 1;
}


void MagnetoRosNode::customPreStep() {

    // &&&
    // + Devel
    if (should_use_rl_) {
        if (((MagnetoInterface*)interface_)->MonitorEndofState()) {
            ready_for_next_step_ = true;
            // ? maybe the request for magnetism modifiers can be made here? I think it can
        }

        if ((ros::Time::now().toSec() - sim_resume_time_) <= resume_duration_) {
            // ROS_INFO_STREAM("Waiting for physics to resolve given duration " << resume_duration_ << "(" << (ros::Time::now().toSec() - sim_resume_time_) << "/" << resume_duration_ << ")");
            std::cout << "Waiting for physics to resolve given duration " << resume_duration_ << "(" << (ros::Time::now().toSec() - sim_resume_time_) << "/" << resume_duration_ << ")" << std::endl;
        } else if (!plugin_updated_ && ready_for_next_step_) {
            ready_for_next_step_ = false;

            if (!ready_for_rl_input_) {
                std::cout << "Waiting for ready_for_rl_update request; simulation is paused..." << std::endl;   
            }
            while (!ready_for_rl_input_) {
                // ROS_INFO_STREAM("Waiting for ready_for_rl_update request; simulation is paused...");
                // std::cout << "Waiting for ready_for_rl_update request; simulation is paused..." << std::endl;
                ros::Duration(0.001).sleep();
            }
            // MagnetoRosNode::pluginInterface();
            MagnetoRosNode::enactActionCommand();
            sim_resume_time_ = ros::Time::now().toSec();
        }

        if (plugin_updated_ && !((MagnetoInterface*)interface_)->MonitorEndofState()){
            plugin_updated_ = false;
            std::cout<<" plugin_updated_  = " << plugin_updated_ << std::endl;            
        }
    }
    // &&&

    static Eigen::VectorXd qInit = robot_->getPositions();

    t_ = (double)count_ * servo_rate_;

    Eigen::VectorXd q = robot_->getPositions();
    Eigen::VectorXd qdot = robot_->getVelocities();

    for(int i=0; i< Magneto::n_adof; ++i) {
        sensor_data_->q[i] = q[Magneto::idx_adof[i]];
        sensor_data_->qdot[i] = qdot[Magneto::idx_adof[i]];
    }

    for(int i=0; i< Magneto::n_vdof; ++i) {
        sensor_data_->virtual_q[i] = q[Magneto::idx_vdof[i]];
        sensor_data_->virtual_qdot[i] = qdot[Magneto::idx_vdof[i]];
    }
    // update contact_distance_
    UpdateContactDistance_();
    // update sensor_data_->b_foot_contact 
    UpdateContactSwitchData_();
    // update sensor_data_->foot_wrench
    UpdateContactWrenchData_();

    // --------------------------------------------------------------
    //          COMPUTE COMMAND - desired joint acc/trq etc
    // --------------------------------------------------------------
    // CheckInterrupt_();
    // ROS_INFO("getCommand start");
    // ROS_INFO_STREAM("q\n" << command_->q);
    // ROS_INFO_STREAM("qdot\n" << command_->qdot);
    // ROS_INFO_STREAM("jtrg\n" << command_->jtrq);
    // TODO figure out whether the command sent is being updated here as desired?
    ((MagnetoInterface*)interface_)->getCommand(sensor_data_, command_);
    // ROS_INFO_STREAM("post q\n" << command_->q);
    // ROS_INFO_STREAM("post qdot\n" << command_->qdot);
    // ROS_INFO_STREAM("post jtrg\n" << command_->jtrq);
    // ROS_INFO("getCommand end");
    
    if (b_plot_result_) {
        bool b_planner = ((MagnetoInterface*)interface_)->IsPlannerUpdated();
        bool b_foot_planner = ((MagnetoInterface*)interface_)->IsFootPlannerUpdated();
        if(b_planner || b_foot_planner) world_->removeAllSimpleFrames();
        if (b_planner) PlotResult_();        
        if (b_foot_planner) PlotFootStepResult_();        
    }
    // --------------------------------------------------------------

    trq_cmd_.setZero();
    // spring in gimbal
    
    // double ks = 1.0;// N/rad
    // for(int i=6; i< Magneto::n_vdof; ++i) {
    //     trq_cmd_[Magneto::idx_vdof[i]] = ks * ( 0.0 - sensor_data_->virtual_q[i]);
    // }

    for(int i=0; i< Magneto::n_adof; ++i) {
        trq_cmd_[Magneto::idx_adof[i]] 
                        = command_->jtrq[i] + 
                          kd_ * (command_->qdot[i] - sensor_data_->qdot[i]) +
                          kp_ * (command_->q[i] - sensor_data_->q[i]);
    }

    // for(int i=0; i< Magneto::n_adof; ++i) {
    //     trq_cmd_[Magneto::idx_adof[i]] 
    //                     = kd_ * (command_->qdot[i] - sensor_data_->qdot[i]) +
    //                       kp_ * (command_->q[i] - sensor_data_->q[i]);
    // }
    
    // for(int i=0; i< Magneto::n_adof; ++i) {
    //     trq_cmd_[Magneto::idx_adof[i]] 
    //                     = command_->jtrq[i] + 
    //                       kd_ * ( - sensor_data_->qdot[i]) +
    //                       kp_ * (command_->q[i] - sensor_data_->q[i]);
    // }

    // for(int i=0; i< Magneto::n_adof; ++i) {
    //     trq_cmd_[Magneto::idx_adof[i]] 
    //                     = command_->jtrq[i];
    // }

    // for(int i=0; i< Magneto::n_adof; ++i) {
    //     trq_cmd_[Magneto::idx_adof[i]] 
    //                     = kd_ * ( 0.0 - sensor_data_->qdot[i]) +
    //                       kp_ * ( qInit[Magneto::idx_adof[i]] - sensor_data_->q[i]);
    // }

    EnforceTorqueLimit();
    ApplyMagneticForce();
    robot_->setForces(trq_cmd_);   

    // SAVE DATA
    //0112 my_utils::saveVector(sensor_data_->alf_wrench, "alf_wrench");
    //0112 my_utils::saveVector(sensor_data_->blf_wrench, "blf_wrench");
    //0112 my_utils::saveVector(sensor_data_->arf_wrench, "arf_wrench");
    //0112 my_utils::saveVector(sensor_data_->brf_wrench, "brf_wrench");

    Eigen::VectorXd trq_act_cmd = Eigen::VectorXd::Zero(Magneto::n_adof);
    for(int i=0; i< Magneto::n_adof; ++i)
        trq_act_cmd[i] = trq_cmd_[Magneto::idx_adof[i]];
    
    //0112 my_utils::saveVector(trq_act_cmd, "trq_fb");
    //0112 my_utils::saveVector(command_->jtrq, "trq_ff");

    //0112 my_utils::saveVector(command_->q, "q_cmd");
    //0112 my_utils::saveVector(sensor_data_->q, "q_sen");

    count_++;
}

void MagnetoRosNode::CheckInterrupt_() {
    // this moved to enableButton
}

void MagnetoRosNode::EnforceTorqueLimit()  {
    for(int i=0; i< n_dof_; ++i) {
        trq_cmd_[i] = trq_cmd_[i] >  trq_lb_[i] ? trq_cmd_[i] : trq_lb_[i];
        trq_cmd_[i] = trq_cmd_[i] <  trq_ub_[i] ? trq_cmd_[i] : trq_ub_[i];
    }    
}

void MagnetoRosNode::ApplyMagneticForce()  {
    bool is_force_local = true;
    bool is_force_global = false;
    Eigen::Vector3d force_w = Eigen::VectorXd::Zero(3);   
    Eigen::Vector3d force = Eigen::VectorXd::Zero(3);   
    Eigen::Vector3d location = Eigen::VectorXd::Zero(3);
    double distance_ratio;
    double distance_constant = contact_threshold_ * 4.; // 0.045
    
    Eigen::Quaternion<double> quat_ground
                            = Eigen::Quaternion<double>( 
                                ground_->getBodyNode("ground_link")
                                        ->getWorldTransform().linear() );
    Eigen::VectorXd p_al = robot_->getBodyNode(MagnetoBodyNode::AL_foot_link)->getWorldTransform().translation(); // &REF
    // std::cout << "Robot position: " << p_al << ", Magnetic force: " << magnetic_force_ << "\n";
    for(auto it : command_->b_magnetism_map) {
        if( it.second ) {
            force[2] = - magnetic_force_;
        } else {
            // distance 0->1 , infinite->0
            distance_ratio = distance_constant / (contact_distance_[it.first] + distance_constant);
            distance_ratio = distance_ratio*distance_ratio;
            force[2] = - distance_ratio*(residual_magnetism_/100.)*magnetic_force_;
            // std::cout<<"res: dist = "<<contact_distance_[it.first]<<", distance_ratio=" << distance_ratio << std::endl;
        }       
        force_w = quat_ground.toRotationMatrix() * force;

        // TODO add in modifier here using service call to env
        // Eigen::VectorXd p_base = robot_->getBodyNode(MagnetoBodyNode::base_link)->getWorldTransform().translation();
        // magneto_rl::MagnetismModifierRequest srv;
        // srv.request.point.x = p_base[0];
        // srv.request.point.y = p_base[1];
        // if (magnetism_update_client_.call(srv)) {
        //     force[2] = force[2] * srv.response.modifier;
        // }
        force[2] = 0.5 * magnetism_modifier_ * force[2];

        // - need to store a magnetism value as a class variable for the node, print out robot_->getBodyNode(it.first) to see what it says
        robot_->getBodyNode(it.first)->addExtForce(force, location, is_force_local);
        // robot_->getBodyNode(it.first)->addExtForce(force_w, location, is_force_global);

        //0112 my_utils::saveVector(force, "force_" + robot_->getBodyNode(it.first)->getName() );
        // std::cout<< robot_->getBodyNode(it.first)->getName().c_str()
        //          << " : " << force_w.transpose() << std::endl;
        // std::cout << "--------------------------" << std::endl;
    }
}

void MagnetoRosNode::PlotResult_() {
    Eigen::VectorXd com_pos = Eigen::VectorXd::Zero(3);

    ((MagnetoInterface*)interface_)-> GetOptimalCoM(com_pos);
    Eigen::Isometry3d com_tf = Eigen::Isometry3d::Identity();
    com_tf.translation() = com_pos;
    std::vector <std::pair<double, Eigen::Vector3d>> feasible_com_list;
    ((MagnetoInterface*)interface_)-> GetFeasibleCoM(feasible_com_list); 

    // set color
    Eigen::Vector4d feasible_color = Eigen::Vector4d(0.0, 0.0, 1.0, 1.0); // blue
    Eigen::Vector4d infeasible_color = Eigen::Vector4d(1.0, 0, 0.0, 1.0); // red

    // set Shape
    std::shared_ptr<dart::dynamics::PointCloudShape> feasible_shape =
        std::make_shared<dart::dynamics::PointCloudShape>(
            dart::dynamics::PointCloudShape(0.003));

    std::shared_ptr<dart::dynamics::PointCloudShape> infeasible_shape =
        std::make_shared<dart::dynamics::PointCloudShape>(
            dart::dynamics::PointCloudShape(0.003));

    std::shared_ptr<dart::dynamics::SphereShape> s_shape =
        std::make_shared<dart::dynamics::SphereShape>(
            dart::dynamics::SphereShape(0.01));


    for (auto it = feasible_com_list.begin(); it != feasible_com_list.end(); it++) {
        if( (*it).first > 0.0 )
            feasible_shape->addPoint((*it).second);
        else
            infeasible_shape->addPoint((*it).second);
    }
    std::cout<<"PlotResult :Feasible Points:" << feasible_shape->getNumPoints()
     << ", Infeasible Points:" << infeasible_shape->getNumPoints() << std::endl;

    // set Frame
    dart::dynamics::SimpleFramePtr feasible_com_frame, infeasible_com_frame;
    dart::dynamics::SimpleFramePtr optimal_com_frame;
    
    feasible_com_frame = std::make_shared<dart::dynamics::SimpleFrame>(
        dart::dynamics::Frame::World(), "com_feasible");
    infeasible_com_frame = std::make_shared<dart::dynamics::SimpleFrame>(
        dart::dynamics::Frame::World(), "com_infeasible");
    optimal_com_frame = std::make_shared<dart::dynamics::SimpleFrame>(
        dart::dynamics::Frame::World(), "com_target", com_tf);
    
    feasible_com_frame->setShape(feasible_shape);
    feasible_com_frame->getVisualAspect(true)->setColor(feasible_color);
    world_->addSimpleFrame(feasible_com_frame);

    infeasible_com_frame->setShape(infeasible_shape);
    infeasible_com_frame->getVisualAspect(true)->setColor(infeasible_color);
    world_->addSimpleFrame(infeasible_com_frame);

    optimal_com_frame->setShape(s_shape);
    optimal_com_frame->getVisualAspect(true)->setColor(infeasible_color);
    world_->addSimpleFrame(optimal_com_frame);

    
}


void MagnetoRosNode::PlotFootStepResult_() {

    world_->removeAllSimpleFrames();
    
    Eigen::VectorXd foot_pos;
    ((MagnetoInterface*)interface_)-> GetNextFootStep(foot_pos);
    Eigen::Isometry3d foot_tf = robot_->getBodyNode("base_link")->getTransform();
    foot_tf.translation() = foot_pos;  
    my_utils::pretty_print(foot_pos, std::cout, "PlotFootStepResult_"); 

    // set shape
    dart::dynamics::BoxShapePtr foot_shape =
        std::make_shared<dart::dynamics::BoxShape>(
            dart::dynamics::BoxShape(Eigen::Vector3d(0.02, 0.02, 0.02))); // Eigen::Vector3d(0.02, 0.02, 0.001)

    // set color
    Eigen::Vector4d foot_step_color = Eigen::Vector4d(0.0, 0.0, 1.0, 1.0); // blue

    // set frame
    dart::dynamics::SimpleFramePtr foot_frame;
    foot_frame = std::make_shared<dart::dynamics::SimpleFrame>(
        dart::dynamics::Frame::World(), "next_foot", foot_tf);
    foot_frame->setShape(foot_shape);
    foot_frame->getVisualAspect(true)->setColor(foot_step_color);
    world_->addSimpleFrame(foot_frame);
}

void MagnetoRosNode::SetParams_() {
    try {
        YAML::Node simulation_cfg;
        if(run_mode_==RUN_MODE::STATICWALK) 
        {
            // simulation_cfg = YAML::LoadFile(THIS_COM "config/Magneto/SIMULATIONWALK.yaml");
            simulation_cfg = YAML::LoadFile(THIS_COM + walkset_);
        }
        else
        {
            std::cout<<"unavailable run mode"<<std::endl;
            exit(0);
        }
            

        my_utils::readParameter(simulation_cfg, "servo_rate", servo_rate_);
        my_utils::readParameter(simulation_cfg["control_configuration"], "kp", kp_);
        my_utils::readParameter(simulation_cfg["control_configuration"], "kd", kd_);
        my_utils::readParameter(simulation_cfg["control_configuration"], "torque_limit", torque_limit_);
        my_utils::readParameter(simulation_cfg, "plot_result", b_plot_result_);
        my_utils::readParameter(simulation_cfg, "motion_script", motion_file_name_);  

        // setting magnetic force
        my_utils::readParameter(simulation_cfg, "magnetic_force", magnetic_force_);  
        my_utils::readParameter(simulation_cfg, "residual_magnetism", residual_magnetism_);  
    } catch (std::runtime_error& e) {
        std::cout << "Error reading parameter [" << e.what() << "] at file: ["
                  << __FILE__ << "]" << std::endl
                  << std::endl;
    }
}

// & This function reads in a yaml file with a series of motions, breaks it up and adds them to a list in magnetointerface
// ? Maybe I can add some functionality to generate a single moveset using some higher-level command, run it, then wait for the next
void MagnetoRosNode::ReadMotions_() {

    std::ostringstream motion_file_name;    
    motion_file_name << THIS_COM << motion_file_name_;

    int num_motion;

    try { 
        YAML::Node motion_cfg = YAML::LoadFile(motion_file_name.str());
        my_utils::readParameter(motion_cfg, "num_motion", num_motion);
        num_initial_steps_ = num_motion;
        for(int i(0); i<num_motion; ++i){
            int link_idx;
            MOTION_DATA md_temp;

            Eigen::VectorXd pos_temp;
            Eigen::VectorXd ori_temp;
            bool is_bodyframe;

            std::ostringstream stringStream;
            stringStream << "motion" << i;
            std::string conf = stringStream.str();    

            my_utils::readParameter(motion_cfg[conf], "foot", link_idx);
            my_utils::readParameter(motion_cfg[conf], "duration", md_temp.motion_period);
            my_utils::readParameter(motion_cfg[conf], "swing_height", md_temp.swing_height);
            my_utils::readParameter(motion_cfg[conf], "pos",pos_temp);
            my_utils::readParameter(motion_cfg[conf], "ori", ori_temp);
            my_utils::readParameter(motion_cfg[conf], "b_relative", is_bodyframe);
            md_temp.pose = POSE_DATA(pos_temp, ori_temp, is_bodyframe);
            // interface_->(WalkingInterruptLogic*)interrupt_
            //             ->motion_command_script_list_
            //             .push_back(MotionCommand(link_idx, md_temp));
            ((MagnetoInterface*)interface_)->AddScriptWalkMotion(link_idx, md_temp); // *NOTE: action trace
        }

    } catch (std::runtime_error& e) {
        std::cout << "Error reading parameter [" << e.what() << "] at file: ["
                  << __FILE__ << "]" << std::endl
                  << std::endl;
    }
}

void MagnetoRosNode::UpdateContactDistance_() {
    // get normal distance from the ground link frame R_ground
    // p{ground} = R_gw * p{world} 
    Eigen::MatrixXd R_ground = ground_->getBodyNode("ground_link")
                                        ->getWorldTransform().linear();
    Eigen::VectorXd p_ground = ground_->getBodyNode("ground_link")
                                        ->getWorldTransform().translation();

    Eigen::MatrixXd R_gw = R_ground.transpose();
    Eigen::MatrixXd p_gw = - R_gw * p_ground;

    Eigen::VectorXd alf = p_gw + R_gw*robot_->getBodyNode("AL_foot_link") // COP frame node?
                            ->getWorldTransform().translation();
    Eigen::VectorXd blf = p_gw + R_gw*robot_->getBodyNode("BL_foot_link")
                            ->getWorldTransform().translation();
    Eigen::VectorXd arf = p_gw + R_gw*robot_->getBodyNode("AR_foot_link")
                             ->getWorldTransform().translation();
    Eigen::VectorXd brf = p_gw + R_gw*robot_->getBodyNode("BR_foot_link")
                             ->getWorldTransform().translation();

    contact_distance_[MagnetoBodyNode::AL_foot_link] = fabs(alf[2]);
    contact_distance_[MagnetoBodyNode::BL_foot_link] = fabs(blf[2]);
    contact_distance_[MagnetoBodyNode::AR_foot_link] = fabs(arf[2]);
    contact_distance_[MagnetoBodyNode::BR_foot_link] = fabs(brf[2]);
}


void MagnetoRosNode::UpdateContactSwitchData_() {
    
    // TODO : distance base -> force base ?  
    sensor_data_->alfoot_contact 
        = contact_distance_[MagnetoBodyNode::AL_foot_link] < contact_threshold_;
    sensor_data_->blfoot_contact
        = contact_distance_[MagnetoBodyNode::BL_foot_link] < contact_threshold_;    
    sensor_data_->arfoot_contact
        = contact_distance_[MagnetoBodyNode::AR_foot_link] < contact_threshold_;
    sensor_data_->brfoot_contact
        = contact_distance_[MagnetoBodyNode::BR_foot_link] < contact_threshold_;

    // static bool first_contact = false;
    // if(!first_contact) {
    //     if(alfoot_contact || blfoot_contact || arfoot_contact || brfoot_contact) {
    //         Eigen::Vector3d p_com = robot_->getCOM();    
    //         std::cout<< count_<< " / first_contact! com position :" << p_com(0) << "," << p_com(1) << "," << p_com(2) << std::endl;

    //         Eigen::Vector3d p_base_link = robot_->getBodyNode("base_link")->getTransform().translation();    
    //         std::cout<< "first_contact! p_base_link position :" << p_base_link(0) << "," << p_base_link(1) << "," << p_base_link(2) << std::endl;

    //         // std::cout<< "wrench" << sensor_data_->alf_wrench(2) << ","  << sensor_data_->arf_wrench(2) << ","  
    //         //                         << sensor_data_->blf_wrench(2) << "," << sensor_data_->brf_wrench(2) << std::endl;
    //         std::cout<< "-------------------    first_contact   ---------------------" << std::endl;
    //         first_contact = true;
    //         // exit(0);
    //     }
    // }

}


void MagnetoRosNode::UpdateContactWrenchData_() {

    // (contact COP link name, contact link node)
    std::vector<std::pair<std::string,dart::dynamics::BodyNode*>> bn_list;

    bn_list.clear();
    bn_list.push_back(std::make_pair("AL_foot_link",robot_->getBodyNode("AL_foot_link_3")));
    bn_list.push_back(std::make_pair("BL_foot_link",robot_->getBodyNode("BL_foot_link_3")));
    bn_list.push_back(std::make_pair("AR_foot_link",robot_->getBodyNode("AR_foot_link_3")));
    bn_list.push_back(std::make_pair("BR_foot_link",robot_->getBodyNode("BR_foot_link_3")));

    // (contact wrench)
    std::vector<Eigen::Vector6d> wrench_local_list;
    std::vector<Eigen::Vector6d> wrench_global_list;  

    wrench_local_list.clear();    
    wrench_global_list.clear();
    for(int i=0; i<bn_list.size(); ++i) {
        wrench_local_list.push_back(Eigen::VectorXd::Zero(6));
        wrench_global_list.push_back(Eigen::VectorXd::Zero(6));
    }

    const dart::collision::CollisionResult& _result =
                            world_->getLastCollisionResult();

    for (const auto& contact : _result.getContacts()) 
    {
        for(int i=0; i<bn_list.size(); ++i) 
        {
            for (const auto& shapeNode :
                bn_list[i].second->getShapeNodesWith<dart::dynamics::CollisionAspect>()) 
            {
                Eigen::VectorXd w_c = Eigen::VectorXd::Zero(6);
                Eigen::Isometry3d T_wc = Eigen::Isometry3d::Identity();

                if ( shapeNode == contact.collisionObject1->getShapeFrame() )               
                    w_c.tail(3) = contact.force;                    
                else if( shapeNode == contact.collisionObject2->getShapeFrame())
                    w_c.tail(3) = - contact.force;                
                else { continue; }
                
                T_wc.translation() = contact.point;
                Eigen::Isometry3d T_wa =
                    robot_->getBodyNode(bn_list[i].first) // Foot COP Frame
                        ->getTransform(dart::dynamics::Frame::World());
                        
                Eigen::Isometry3d T_ca = T_wc.inverse() * T_wa;
                Eigen::Isometry3d T_cw = T_wc.inverse();

                Eigen::MatrixXd AdT_ca = dart::math::getAdTMatrix(T_ca);
                Eigen::MatrixXd AdT_cw = dart::math::getAdTMatrix(T_cw);

                Eigen::VectorXd w_a = AdT_ca.transpose() * w_c;
                Eigen::VectorXd w_w = AdT_cw.transpose() * w_c;
                
                wrench_local_list[i] += w_a;
                wrench_global_list[i] += w_w;  

            }  
        }      
    }

    // std::cout<<"-------------------------------" << std::endl;
   
    // for(int i=0; i<wrench_local_list.size(); ++i)
    // {
    //     std::string printname = bn_list[i].first;
    //     // std::ostream printname;
    //     // printname << bn_list[i].first;
    //     Eigen::VectorXd wrench = wrench_local_list[i];
    //     my_utils::pretty_print(wrench, std::cout, "wrench");
    //     wrench = wrench_global_list[i];
    //     my_utils::pretty_print(wrench, std::cout, "wrench_w");
    // }
    // std::cout<<"-------------------------------" << std::endl;

    sensor_data_->alf_wrench = wrench_local_list[0];
    sensor_data_->blf_wrench = wrench_local_list[1];
    sensor_data_->arf_wrench = wrench_local_list[2];
    sensor_data_->brf_wrench = wrench_local_list[3];
    
}
