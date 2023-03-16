# kn5-obj-converter

Script to convert .kn5 files (Assetto Corsa) to .obj

# Introduction

This is a Python script to convert Assetto Corsa .kn5 files to .obj and optionally ASCII .fbx.
The output .obj files can be opened in any editor, such as Blender, or in game engines, such as Unreal Engine.

# How To

In `convert.py` set the path to your model and run the script as follows:

```bash
$ python convert.py
```

# TODO

- [ ] Implement a simple CLI to allow the user to specify the path to the model
- [ ] Implement the possibility to utilize the `models.ini` file

# Credits

This project was ported from [RaduMC/kn5-converter](https://github.com/RaduMC/kn5-converter/) to Python using ChatGPT for the bulk of the code conversion.
