#  __author: Mark Azer
#  ie3 TU Dortmund

import xml.etree.ElementTree as Et
from random import randrange
import uuid
from uuid import UUID


class Product:
    def __init__(self, name, product_uuid: UUID):
        self.name = name
        self.sub_products = []
        self.items_completed = False
        self.product_finished = False
        self.product_uuid = product_uuid

    def add_sub_product(self, sub_product):
        self.sub_products.append(sub_product)


class SubProduct:
    def __init__(self, option_id, name, description, product_uuid: UUID):
        """

        :param option_id:
        :param name:
        :param description:
        """
        self.name = name
        self.machine_sequence = []
        self.machine_sequence_int = []
        self.machine_time_sequence = []
        self.item_completed = False
        self.name = name
        self.id = option_id
        self.product_uuid = product_uuid
        self.sub_product_job = None
        self.create_sequence(description)

    def create_sequence(self, description):
        """
        this method translates the product requirements into machine sequence
        :param description:
        :return:
        """
        data = description.split()
        self.machine_sequence = [x for x in data[0::2]]
        self.machine_sequence_int = [1, 2, 3, 0]
        self.machine_time_sequence = [int(x) for x in data[1::2]]


class ProductsSimulation:
    def __init__(self, products_xml):
        self.products_tree = Et.parse(products_xml)
        self.products = []
        self.jobs = 0

    def create_products(self, number_of_products):
        """
        this method creates products from the xml file
        :param number_of_products:
        :return:
        """
        self.jobs = number_of_products
        root = self.products_tree.getroot()

        for i in range(number_of_products):
            product_uuid = uuid.uuid1()
            product_options = len(list(root))
            random_option_idx = randrange(product_options)
            random_option = root[random_option_idx]
            product_name = random_option.attrib["Name"]
            new_product = Product(product_name, product_uuid)
            for part in random_option.iter('Part'):
                part_options = len(list(part))
                random_option_idx = randrange(part_options)
                random_option = part[random_option_idx]
                option_id = random_option.find('ID').text
                part_name = part.attrib["Name"] + '_' + str(i + 1)
                description = random_option.find('Description').text
                new_sub_product = SubProduct(option_id, part_name, description, product_uuid)
                new_product.add_sub_product(new_sub_product)
            self.products.append(new_product)
