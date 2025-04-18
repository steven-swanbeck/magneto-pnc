#pragma once

#include <my_pnc/InterruptLogic.hpp>

// Forward Declare Control Architecture
class MagnetoControlArchitecture;
class MagnetoStateProvider;
class MotionCommand;

class WalkingInterruptLogic : public InterruptLogic {
 public:
  WalkingInterruptLogic(MagnetoControlArchitecture* ctrl_arch_);
  ~WalkingInterruptLogic();

  void processInterrupts();

  MagnetoControlArchitecture* ctrl_arch_;
  MagnetoStateProvider* sp_;

  std::deque<MotionCommand> motion_command_script_list_;
  MOTION_DATA motion_data_default_;
  MotionCommand* motion_command_alfoot_;
  MotionCommand* motion_command_blfoot_;
  MotionCommand* motion_command_arfoot_;
  MotionCommand* motion_command_brfoot_;
  MotionCommand* motion_command_instant_;

  void executeCommand();
};
