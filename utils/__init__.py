from jinja2 import Environment, FileSystemLoader, StrictUndefined
import yaml


def render_xml(env=Environment(loader=FileSystemLoader(searchpath="./"),
                               trim_blocks=False,
                               lstrip_blocks=False,
                               undefined=StrictUndefined, ), source_file="./", template_file="./", output_file="./"):
    with open(source_file, "r") as file:
        try:
            source_data = yaml.safe_load(file)
        except yaml.YAMLError as err:
            print(err)

    try:
        template = env.get_template(name=template_file)
        xml_out_data = template.render(source_data=source_data)
        with open(output_file, "w") as file:
            file.write(xml_out_data)
    except Exception as err:
        print(err)

    return source_data
