python3 md2nb.py index.md
jupyter nbconvert index.ipynb --to slides
mv index.slides.html index.html
open index.html