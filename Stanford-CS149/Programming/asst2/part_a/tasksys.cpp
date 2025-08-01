#include "tasksys.h"
#include <thread>
#include <atomic>
#include <mutex>
#include <iostream>
#include <string>

IRunnable::~IRunnable() {}

ITaskSystem::ITaskSystem(int num_threads) {}
ITaskSystem::~ITaskSystem() {}
/* #region Part A */

/*
 * ================================================================
 * Serial task system implementation
 * ================================================================
 */
/* #region */
const char *TaskSystemSerial::name()
{
    return "Serial";
}

TaskSystemSerial::TaskSystemSerial(int num_threads) : ITaskSystem(num_threads)
{
}

TaskSystemSerial::~TaskSystemSerial() {}

void TaskSystemSerial::run(IRunnable *runnable, int num_total_tasks)
{
    for (int i = 0; i < num_total_tasks; i++)
    {
        runnable->runTask(i, num_total_tasks);
    }
}

TaskID TaskSystemSerial::runAsyncWithDeps(IRunnable *runnable, int num_total_tasks,
                                          const std::vector<TaskID> &deps)
{
    // You do not need to implement this method.
    return 0;
}

void TaskSystemSerial::sync()
{
    // You do not need to implement this method.
    return;
}
/* #endregion */

/*
 * ================================================================
 * Parallel Task System Implementation
 * ================================================================
 */
/* #region  */
const char *TaskSystemParallelSpawn::name()
{
    return "Parallel + Always Spawn";
}

TaskSystemParallelSpawn::TaskSystemParallelSpawn(int num_threads) : ITaskSystem(num_threads)
{
    //
    // TODO: CS149 student implementations may decide to perform setup
    // operations (such as thread pool construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //

    this->num_threads = num_threads;
    this->workers = new std::thread[num_threads];
}

TaskSystemParallelSpawn::~TaskSystemParallelSpawn() {}

void TaskSystemParallelSpawn::run(IRunnable *runnable, int num_total_tasks)
{
    //
    // TODO: CS149 students will modify the implementation of this
    // method in Part A.  The implementation provided below runs all
    // tasks sequentially on the calling thread.
    //

    int tasks_per_thread = (num_total_tasks + num_threads - 1) / num_threads;
    for (int i = 0; i < num_threads; i++)
    {
        // assign work for thread
        auto work = [=](int start, int end)
        {
            for (int task = start; task < end; task++)
            {
                runnable->runTask(task, num_total_tasks);
            }
        };

        workers[i] = std::thread(
            work,
            i * tasks_per_thread,
            std::min((i + 1) * tasks_per_thread, num_total_tasks));
    }

    // join threads
    for (int i = 0; i < num_threads; i++)
    {
        workers[i].join();
    }

    return;
}

TaskID TaskSystemParallelSpawn::runAsyncWithDeps(IRunnable *runnable, int num_total_tasks,
                                                 const std::vector<TaskID> &deps)
{
    // You do not need to implement this method.
    return 0;
}

void TaskSystemParallelSpawn::sync()
{
    // You do not need to implement this method.
    return;
}

/* #endregion */

/*
 * ================================================================
 * Parallel Thread Pool Spinning Task System Implementation
 * ================================================================
 */
/* #region  */
const char *TaskSystemParallelThreadPoolSpinning::name()
{
    return "Parallel + Thread Pool + Spin";
}

TaskSystemParallelThreadPoolSpinning::TaskSystemParallelThreadPoolSpinning(int num_threads) : ITaskSystem(num_threads)
{
    //
    // TODO: CS149 student implementations may decide to perform setup
    // operations (such as thread pool construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //
    this->num_threads = num_threads;
    this->workers = new std::thread[this->num_threads];
    this->mu = new std::mutex();
    this->cond = new std::condition_variable();

    this->destruct = false;
    this->curr_task = new std::atomic<int>(0);
    this->completed_tasks = new std::atomic<int>(0);
    this->total_tasks = 0;

    for (int i = 0; i < num_threads; i++)
    {
        workers[i] = std::thread(&TaskSystemParallelThreadPoolSpinning::spin, this);
    }
}

TaskSystemParallelThreadPoolSpinning::~TaskSystemParallelThreadPoolSpinning()
{
    this->destruct = true;
    for (int i = 0; i < this->num_threads; i++)
    {
        workers[i].join();
    }
}

void TaskSystemParallelThreadPoolSpinning::spin()
{
    while (true)
    {
        if (this->destruct)
        {
            break;
        }

        std::unique_lock<std::mutex> l(*mu);
        if (this->curr_task->load() < this->total_tasks)
        {
            int task_to_run = this->curr_task->operator++() - 1;
            l.unlock();
            this->runnable->runTask(task_to_run, this->total_tasks);
            if (this->completed_tasks->operator++() == this->total_tasks)
            {
                this->cond->notify_all();
            }
        }
    }
}

void TaskSystemParallelThreadPoolSpinning::run(IRunnable *runnable, int num_total_tasks)
{

    //
    // TODO: CS149 students will modify the implementation of this
    // method in Part A.  The implementation provided below runs all
    // tasks sequentially on the calling thread.
    //

    // reset trackers
    std::unique_lock<std::mutex> lock(*this->mu);
    this->completed_tasks->store(0);
    this->curr_task->store(0);
    this->total_tasks = num_total_tasks;
    // queue work
    this->runnable = runnable;
    this->cond->wait(lock);
}

TaskID TaskSystemParallelThreadPoolSpinning::runAsyncWithDeps(IRunnable *runnable, int num_total_tasks,
                                                              const std::vector<TaskID> &deps)
{
    // You do not need to implement this method.
    return 0;
}

void TaskSystemParallelThreadPoolSpinning::sync()
{
    // You do not need to implement this method.
    return;
}

/* #endregion */

/* #endregion */
/*
 * ================================================================
 * Parallel Thread Pool Sleeping Task System Implementation
 * ================================================================
 */
/* #region  */
const char *TaskSystemParallelThreadPoolSleeping::name()
{
    return "Parallel + Thread Pool + Sleep";
}

TaskSystemParallelThreadPoolSleeping::TaskSystemParallelThreadPoolSleeping(int num_threads) : ITaskSystem(num_threads)
{
    // waiting_threads = new std::atomic<int>(0);
    waiting_threads.store(0);

    // workers
    this->num_threads = num_threads;
    workers = new std::thread[this->num_threads];

    // spin up threads
    for (int i = 0; i < num_threads; i++)
    {
        workers[i] = std::thread(&TaskSystemParallelThreadPoolSleeping::spin, this);
    }
}

TaskSystemParallelThreadPoolSleeping::~TaskSystemParallelThreadPoolSleeping()
{
    // indicate destruction and unblock threads so they can break
    destruct = true;
    new_work_assigned.notify_all();

    for (int i = 0; i < num_threads; i++)
    {
        workers[i].join();
    }
}

/// @brief Spins up a new thread.
void TaskSystemParallelThreadPoolSleeping::spin()
{
    while (true)
    {
        if (destruct)
        {
            break;
        }

        std::unique_lock<std::mutex> lock(mu);
        // if there are more tasks unassigned
        if (curr_task < total_tasks)
        {
            // assign the current value of curr_task
            int task_to_run = curr_task++;
            lock.unlock();

            // run the assigned task
            runnable->runTask(task_to_run, total_tasks);
        }
        else
        {
            waiting_threads.operator++();
            finished_mu.lock();
            finished_mu.unlock();
            finished.notify_all();

            new_work_assigned.wait(lock);
        }
    }
}

void TaskSystemParallelThreadPoolSleeping::run(IRunnable *runnable, int num_total_tasks)
{
    std::unique_lock<std::mutex> lock(mu);

    // reset current task info
    waiting_threads.store(0);
    curr_task = 0;
    this->total_tasks = num_total_tasks;

    // queue work and tell threads to start working
    this->runnable = runnable;

    lock.unlock();

    std::unique_lock<std::mutex> finished_lock(finished_mu);
    new_work_assigned.notify_all();

    finished.wait(finished_lock, [&]()
                         { return waiting_threads.load() == this->num_threads; });
}

TaskID TaskSystemParallelThreadPoolSleeping::runAsyncWithDeps(IRunnable *runnable, int num_total_tasks,
                                                              const std::vector<TaskID> &deps)
{

    //
    // TODO: CS149 students will implement this method in Part B.
    //

    return 0;
}

void TaskSystemParallelThreadPoolSleeping::sync()
{

    //
    // TODO: CS149 students will modify the implementation of this method in Part B.
    //

    return;
}

/* #endregion */