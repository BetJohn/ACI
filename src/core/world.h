#pragma once

#include "window/input_controller.h"


class World : public InputController
{
 public:
    World();
    virtual ~World() {}
    virtual void Init() {}
    virtual void FrameStart() {}
    virtual void Update(float deltaTimeSeconds) {}
    virtual void FrameEnd() {}

    virtual void Run();
    void Pause();
    virtual void Exit();

    double GetLastFrameTime();

 protected:
    void ComputeFrameDeltaTime();
protected:
    virtual void LoopUpdate(); 

 private:
    double previousTime;
    double elapsedTime;
    bool paused;
protected:
    double deltaTime;
    bool shouldClose;
};
