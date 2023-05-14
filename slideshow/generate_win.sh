python .\md2nb.py index.md
jupyter nbconvert index.ipynb --to slides
Rename-Item index.slides.html index.html
Invoke-Item index.html

# ----.bat
# ```
# python md2nb.py index.md
# jupyter nbconvert index.ipynb --to slides
# ren index.slides.html index.html
# start index.html
# ```