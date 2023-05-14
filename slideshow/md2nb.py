import nbformat as nbf
import re
import argparse

parser = argparse.ArgumentParser(description='Convert markdown to notebook')
parser.add_argument('file_name', metavar='file_name', type=str, default='index.md')

args = parser.parse_args()

slidemeta = {"slideshow": {"slide_type": "slide"}}

prenb = nbf.v4.new_notebook()
name = args.file_name.split('.')[0]

with open(args.file_name) as f:
    text = f.read()
    texts = re.split(r'-{4,}', text)
    prenb['cells'] = [nbf.v4.new_markdown_cell(text) for text in texts]
    for cell in prenb['cells']:
        cell.metadata = slidemeta

with open(f'{name}.ipynb', 'w') as f:
    nbf.write(prenb, f)