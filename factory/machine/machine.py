#  __author: Mark Azer
#  ie3 TU Dortmund
import uuid
import xml.etree.ElementTree as Et
from random import randrange
from manpy.simulation.imports import MachineJobShop, QueueJobShop, ExitJobShop, Job


class Machine:
    def __init__(self, machine_uuid, machine_type, machine_power):
        self.type = machine_type
        self.machine_uuid = machine_uuid
        self.machine_power = machine_power


class Factory:
    def __init__(self):
        self.machines = 4


class MachinesSimulation:
    def __init__(self, machines_xml):
        self.machines = []
        self.machine_type_list = []
        self.machines_tree = Et.parse(machines_xml)
        self.manpy_queue_list = []
        self.manpy_machine_list = []

    def create_machines(self, max_machine_redundancy):
        """
        this method creates the machines in a factory with at least one of each machine type
        the machines specs are fetched from xml file
        :param max_machine_redundancy: the maximum number of machines of the same type
        :return:
        """
        root = self.machines_tree.getroot()
        self.manpy_queue_list.append(
            QueueJobShop('assembler Queue', 'assembler Queue', capacity=float("inf"), schedulingRule="Priority"))
        self.manpy_machine_list.append(
            MachineJobShop('assembler Machine', 'assembler Machine'))

        for machine in root.iter('Machine'):
            machine_type = machine.attrib["Name"]
            machine_power = machine.find('Specifications').find('Electrical').find('Power').text
            self.manpy_queue_list.append(
                QueueJobShop(machine_type + " Queue", machine_type + " Queue", capacity=float("inf"),
                             schedulingRule="Priority"))
            self.machine_type_list.append(machine_type)
            for i in range(randrange(max_machine_redundancy) + 1):
                machine_uuid = uuid.uuid1()
                new_machine = Machine(str(machine_uuid), machine_type, int(machine_power))
                self.machines.append(new_machine)
                self.manpy_machine_list.append(
                    MachineJobShop(str(machine_uuid), machine_type + " Machine"))
