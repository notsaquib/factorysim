from factory.factorySim import FactorySim
from factorygym.factoryGymA import FsimEnv
from utils import render_xml
import argparse

from utils.configs import load_configs, set_logging_parameters


def main(project_configs):
    products_xml = './product/products.xml'
    machines_xml = './factory/machine/machines.xml'
    if project_configs["general"]["render_xml"]:
        render_xml(source_file='./product/products.yml', template_file='./product/products.jinja',
                   output_file=products_xml)
        render_xml(source_file='./factory/machine/machines.yml', template_file='./factory/machine/machines.jinja',
                   output_file=machines_xml)

    env = FsimEnv(5, 1, './product/products.xml', './factory/machine/machines.xml', './energy/day_ahead.csv')
    state = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        for i in range(3):
            state, reward, done, info = env.step(action)
    env.render()
    state = env.reset()
    done = False
    while not done:
        action = [0, 0, 0, 0, 0]
        state, reward, done, info = env.step(action)
    env.render()
    env.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--release',
        help="release flag",
        action="store_true",
        default=False
    )
    parser.add_argument("--configs", help="config.json file", default='config.json')
    args = parser.parse_args()
    configs = load_configs(args.configs)
    set_logging_parameters(configs, args.release)
    main(configs)
