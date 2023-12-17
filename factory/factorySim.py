import random

import numpy as np

from product.product import ProductsSimulation
from factory.machine.machine import MachinesSimulation
from manpy.simulation.imports import ExitJobShop, Job, ShiftScheduler
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from energy.energy import EnergyHandler


class FactorySim:
    def __init__(self):
        self.current_time_min = 0
        self.action = None
        self.energy_handler = None
        self.energy_price = 0
        self.energy_price_list = [0]
        self.product_simulator: ProductsSimulation = None
        self.machine_simulator: MachinesSimulation = None
        self.machine_list = None
        self.exit = [ExitJobShop('E', 'Exit')]
        self.job_list = []
        self.to_assembler_list = []
        self.finished_job_list = []
        self.current_time = pd.Timestamp.now()
        self.df = pd.DataFrame()

    def create_products(self, products_xml, number_of_products):
        self.product_simulator = ProductsSimulation(products_xml)
        self.product_simulator.create_products(number_of_products)

    def create_machines(self, machines_xml, max_machine_redundancy):
        self.machine_simulator = MachinesSimulation(machines_xml)
        self.machine_simulator.create_machines(max_machine_redundancy)

    def create_energy_market(self, market_csv):
        self.energy_handler = EnergyHandler()
        self.energy_handler.load_market(market_csv)

    def create_jobs(self):
        '''
        this method translates the machine sequence of products into manpy jobs
        :return:
        '''
        j = 0
        for product in self.product_simulator.products:
            for item in product.sub_products:
                sequence = item.machine_sequence
                sequence_time = item.machine_time_sequence
                seq = []
                for i in range(len(sequence)):

                    new_queue = None
                    new_machine = None
                    for queue in self.machine_simulator.manpy_queue_list:
                        if queue.name == sequence[i] + " Queue":
                            new_queue = queue.id
                    for machine in self.machine_simulator.manpy_machine_list:
                        if machine.name == sequence[i] + " Machine":
                            new_machine = machine.id
                    seq.append({"stationIdsList": [new_queue]})
                    seq.append(
                        {"stationIdsList": [new_machine], "processingTime": {"Fixed": {"mean": sequence_time[i]}}})
                seq.append({"stationIdsList": ["E"]})
                new_job = Job(str(item.product_uuid), item.name, route=seq, priority=j)
                item.sub_product_job = new_job
                self.job_list.append(new_job)
                j = j + 1

    def run(self, test=0):
        '''
        this method runs the scheduling problem
        :param test:
        :return:
        '''
        for machine in self.machine_simulator.manpy_machine_list:
            machine.initialize()
        # add all the objects in a list
        object_list = self.machine_simulator.manpy_machine_list + self.machine_simulator.manpy_queue_list + self.exit + self.job_list
        print(object_list)
        # set the length of the experiment
        max_sim_time = float('inf')
        # call the runSimulation giving the objects and the length of the experiment

        for i in range(10000):
            self.step()
            # print(i)
            # time.sleep(0.25)
            # runSimulation(object_list, max_sim_time)

    def step(self, action):
        '''
        this method steps 1 time step into the schedule
        :return:
        '''
        self.action = action
        SS = []
        for i in range(len(action)):
            if self.machine_simulator.manpy_machine_list[i].isProcessing == False and action[i]:
                SS.append(ShiftScheduler(victim=self.machine_simulator.manpy_machine_list[i], shiftPattern=[[2, 5], ]))
        object_list = self.machine_simulator.manpy_machine_list + self.machine_simulator.manpy_queue_list + self.exit + self.job_list + SS
        step_time = float(1)
        from manpy.simulation.Globals import runSimulation
        runSimulation(object_list, step_time)
        for job in self.job_list:
            for item in job.schedule[-6:]:
                if item['entranceTime'] == 0.0:
                    item['entranceTime'] = self.current_time_min
                if 'exitTime' in item and item['exitTime'] == 0.0:
                    item['exitTime'] = self.current_time_min
            if len(job.schedule) != 0:
                current_station = job.schedule[-1]
                if not ('exitTime' in current_station):
                    if not (current_station['station'].id == 'E'):
                        job.route = job.route[-len(job.remainingRoute) - 1:]
                        if 'processingTime' in job.route[0]:
                            remaining_time = job.route[0]['processingTime']['Fixed']['mean'] - 1
                            job.route[0]['processingTime'] = {'Fixed': {'mean': remaining_time}}
                    else:
                        self.to_assembler_list.append(job)
                        self.job_list.remove(job)
                        self.assemble()

        self.current_time_min += 1
        price = self.energy_handler.get_price_by_time(
            pd.to_timedelta(self.current_time_min, unit='m') + self.current_time)
        self.energy_price_list.append(price)
        return self.get_observation()

    def get_observation(self):
        if self.action is None:
            action = np.zeros(len(self.machine_simulator.manpy_machine_list))
        else:
            action = self.action
        machines_state = []
        products_state = []
        reward_fut = []
        for machine in self.machine_simulator.manpy_machine_list:
            machine_name = machine.name.split(' ')[0]
            machine_state = [machine.isProcessing]
            product_state = [0, 0, 0, 0, 0]
            remaining_time = 0
            if machine.isProcessing:
                dueDate = machine.currentEntity.dueDate
                priority = machine.currentEntity.priority
                try:
                    remaining_time = machine.currentEntity.route[0]['processingTime']['Fixed']['mean']
                except:
                    remaining_time = machine.currentEntity.route[1]['processingTime']['Fixed']['mean']
                product_state = [dueDate, priority, remaining_time, self.energy_price_list[-1], self.current_time_min]
            reward_fut.append(remaining_time * self.energy_price_list[-1])
            for queue in self.machine_simulator.manpy_queue_list:
                if queue.name.split(' ')[0] == machine_name:
                    if hasattr(queue,"isProcessing"):
                        machine_state.append(queue.isProcessing)
                    else:
                        machine_state.append(False)
                    break
            machines_state.append(machine_state)
            products_state.append(product_state)
        actions_state = 1 - np.array(action)
        reward = np.ones(len(self.machine_simulator.manpy_machine_list)) * self.calculate_price() + reward_fut
        obs_space = {}
        for i in range(len(self.machine_simulator.manpy_machine_list)):
            observation = {"queue_state": machines_state[i], "action_state": actions_state[i],
                           "current_job_state": products_state[i]}
            obs_space[f"agent{i}"] = observation

        return [obs_space, reward, 0, 0]

    def assemble(self):
        '''
        this method checks if all the sub-products of a product are complete and creates an assembly job
        :return:
        '''
        for product in self.product_simulator.products:
            assemble_ready = True
            for sub_product in product.sub_products:
                if not (sub_product.sub_product_job in self.to_assembler_list):
                    assemble_ready = False

            if assemble_ready:
                for sub_product in product.sub_products:
                    self.finished_job_list.append(sub_product.sub_product_job)
                    self.to_assembler_list.remove(sub_product.sub_product_job)
                assemble_job_route = [{"stationIdsList": ["assembler Queue"]}, {"stationIdsList": ["assembler Machine"],
                                                                                "processingTime": {
                                                                                    'Fixed': {'mean': 10}}},
                                      {"stationIdsList": ["E"]}]
                i_name = int(sub_product.sub_product_job.name.split('_')[1])
                self.job_list.append(
                    Job(product.name + "_" + str(i_name), product.name + "_" + str(i_name), route=assemble_job_route,
                        priority=i_name))

    def calculate_price(self):
        current_time = self.current_time
        self.energy_price=0
        j = 0
        price = 0
        for J in self.finished_job_list:
            for record in J.schedule:
                if not (record["station"].objName in ['assembler Queue', 'Exit', 'Milling Queue', 'Turning Queue',
                                                      'Machining Queue',
                                                      'Drilling Queue']):
                    if j == 0:
                        ent_time = record["entranceTime"]
                        price = self.energy_price_list[ent_time]
                        j = 1

                    if 'exitTime' in record:
                        power = 9999999999999
                        exit_time = record['exitTime']
                        machine_uuid = record['station'].id
                        for machine in self.machine_simulator.machines:
                            if machine.machine_uuid == machine_uuid:
                                power = machine.machine_power
                        for time_step in range(exit_time - ent_time):
                            if time_step % 60 == 0:
                                price = self.energy_price_list[ent_time + time_step]
                            self.energy_price = self.energy_price + price * (power / 60) / 1000
        #print(self.energy_price)
        return self.energy_price


    def render(self):
        current_time = self.current_time
        self.calculate_price()
        print(self.energy_price)
        return self.energy_price
        r = lambda: random.randint(0, 255)

        colors = []
        # print the results
        j = 0
        for J in self.to_assembler_list + self.finished_job_list:
            for record in J.schedule:
                if not (record["station"].objName in ['assembler Queue', 'Exit', 'Milling Queue', 'Turning Queue',
                                                      'Machining Queue',
                                                      'Drilling Queue']):
                    if j == 0:
                        ent_time = record["entranceTime"]
                        j = 1

                    if 'exitTime' in record:
                        self.df = pd.concat([self.df, pd.DataFrame(dict(Task=record["station"].objName,
                                                                        Start=pd.to_timedelta(ent_time,
                                                                                              unit='m') + current_time,
                                                                        Finish=pd.to_timedelta(record['exitTime'],
                                                                                               unit='m') + current_time,
                                                                        Resource=J.name), index=[0])])
                        j = 0
                        colors.append('#%02X%02X%02X' % (r(), r(), r()))

        # fig = px.timeline(self.df, x_start="Start", x_end="Finish", y="Task", color="Resource")
        # fig.show()
        from plotly.subplots import make_subplots

        fig_sub = make_subplots(rows=2, shared_xaxes=True, row_heights=[0.9, 0.1],
                                subplot_titles=['Schedule', 'Energy Prices'])
        for i in range(len(px.timeline(self.df, x_start="Start", x_end="Finish", y="Task", color="Resource")['data'])):
            fig_sub.add_trace(
                px.timeline(self.df, x_start="Start", x_end="Finish", y="Task", color="Resource")['data'][i], row=1,
                col=1)
        # for j in range(len(px.timeline(self.df, x_start="Start", x_end="Finish", y="Resource", color="Task")['data'])):
        #    fig_sub.add_trace(px.timeline(self.df, x_start="Start", x_end="Finish", y="Resource", color="Task")['data'][j], row=2, col=1)
        eneregy_fig = self.energy_handler.plot_prices()
        fig_sub.add_trace(eneregy_fig, row=2, col=1)
        fig_sub.update_xaxes(type='date')
        fig_sub.show()
        self.calculate_price()
        print("price of schedule in episode: " + self.energy_price)
