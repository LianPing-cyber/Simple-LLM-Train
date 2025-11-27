from jinja2 import Environment, FileSystemLoader
import os

template_folder = os.path.dirname(__file__)

env = Environment(loader=FileSystemLoader(template_folder))

def get_template(template_name):
    return env.get_template(template_name + ".md.jinja")