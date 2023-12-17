import numpy as np

from factory.factoryEnv import JssEnv
from factory.machine.machine import Factory
from product.product import ProductsSimulation
import random


def main(name):
    product_sim = ProductsSimulation('./product/products.xml')
    product_sim.create_products(4)
    factory = Factory()
    env = JssEnv(product_sim, factory)
    env.reset()
    done = False
    print(env.needed_machine_jobs)
    remaining_jobs = np.array([x for x in range(8)])
    while not done:
        print(env.needed_machine_jobs)

        # if we haven't performed any action, we go to the next time step
        no_op = True
        remaining_jobs_m = remaining_jobs[env.needed_machine_jobs != -1]
        for action_to_do in remaining_jobs_m:
#        action_to_do = random.choice(remaining_jobs_m)
           for machine in range(env.machines):

                if done:
                    break
                if env.machine_legal[machine]:

                    if (
                            env.needed_machine_jobs[action_to_do] == machine and env.legal_actions[action_to_do]
                    ):
                        print(env.current_time_step)
                        no_op = False

                        state, reward, done, _ = env.step(action_to_do)

        if no_op and not done:
            previous_time_step = env.current_time_step
            env.increase_time_step()

    env.render()
    print(env.last_time_step)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
