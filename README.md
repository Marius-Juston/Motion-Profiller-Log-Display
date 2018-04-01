# Motion Profiller Log Display
Displays the Walton Robotics Log files as graphs for simple and quick troubleshooting

## Why to use this
### Its Use
PID tuning is hard to do and visualizing what the robot was meant to do compared to what the robot actually
did is something that is very helpful for troubleshooting problems.
This program is meant to visualise the logs created by Walton Robotics
<a href=https://github.com/ThundrHawk/Actually-Simple-Splines>Actually-Simple-Splines</a> repository which
includes a custom PID loop and motions that are easy to utilize.
### What it can do
This program allows for the visualization the actual path the robot took, the path the robot should have taken, the
velocity it was at, the velocity it should have been at, the lag error, angle error and the cross track error during
the motion and the left and right motor powers. This program also allows you to find some PID constants.
PID constants are used to tune the feedback loop in order to make the motion as accurate as possible. Finding the
constants usually takes a lot of trial and errors but thanks to this program you can find 3 constants (kV, kK and kAcc).


