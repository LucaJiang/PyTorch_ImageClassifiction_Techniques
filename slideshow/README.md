# Instruction on how to use markdown to generate slides

## Write content in [index.md](index.md)

Use this notation to split slides:

```md
--------------------
```

More than 4 `-` will be treated as a slide separator.

## Convert markdown to slides html

Run the generate.sh under this directory

```sh
sh generate.sh
``` 

> This generate.sh is only tested on macos.
> For windows user, you can convert it into powershell script or bat file.

If success, the presentation will be opened automatically in your browser. 

## View online

You can view the presentation online at https://lucajiang.github.io/PyTorch_ImageClassifiction_Techniques/slideshow/
