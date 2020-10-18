from src.envs.scheduling.scheduling_envs.identical_machines import IdenticalMachines, Action


def tst_identical_machines():
    jobs_lengths = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    num_machines = 2
    maximal_episode = 5
    rnd = 2

    identical_machines = IdenticalMachines(jobs_lengths=jobs_lengths,
                                           num_machines=num_machines,
                                           maximal_episode=maximal_episode,
                                           rnd=rnd)
    print("Start")
    identical_machines.reset()
    print(identical_machines.render())
    print(f"possible actions={str(identical_machines.get_possible_actions())}")

    print("Let's move 0.5 to the other queue")
    state_nxt, reward, done, info = identical_machines.step(Action(from_machine=0, job_idx=2, to_machine=1))
    print(identical_machines.render())
    print(f"state_next=\n{state_nxt}\nreward={reward}\ndone={done}\ninfo={info}")
    print(f"possible actions={identical_machines.get_possible_actions()}")

    for _ in range(100):
        print(f"state is:\n{identical_machines.queues}")
        action = identical_machines.get_random_action()
        print(f"action is:\n{action}")
        state_nxt, reward, done, info = identical_machines.step(action)
        print(f"reward={reward}, done={done}")
        if done:
            print(f"state_nxt={identical_machines.queues}")
            print(f"Doing resent")



if __name__ == "__main__":
    tst_identical_machines()