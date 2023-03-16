# Assetto Corsa .kn5 to .obj Converter

## Introduction

This is a Python script to convert Assetto Corsa .kn5 files to .obj and optionally ASCII .fbx.
The output .obj files can be opened in any editor, such as Blender, or in game engines, such as Unreal Engine.

## How To

The script has a simple command line interface, which requires the path to the model directory as argument. The converter can be used as follows:

```bash
$ python convert.py ./path/to/model
```

Use the following command to view the CLI options:

```bash
$ python convert.py -h
```

## Credits

This project is a Python port of [RaduMC/kn5-converter](https://github.com/RaduMC/kn5-converter/).
ChatGPT was used to do the bulk of the conversion automatically. The project was ported to Python to make more convenient to use cross-platform.

### Notable Changes:

- Included alpha transparancy in the .obj based on the primary texture alpha channel
