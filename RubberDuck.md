# RubberDuck
![](./Data/PIC/O.jpg)
# Table of content
- [Introduction](#introduction)
- [Uploading Dataset to github Problem](#uploading-dataset-to-github-problem)
- [Different image size and different image file type](#different-image-size-and-different-image-file-type)
- [Calling  modules from other folder](#calling-modules-from-other-folder)
- [Testing some files that don't return anything](#testing-some-files-that-dont-return-anything)
- [Cannot load 2 different ways of loading at the same time](#cannot-load-2-different-ways-of-loading-at-the-same-time)

# Introduction

Rubber duck debugging, also known as rubber ducking, is a problem-solving method where the person explains their problem to an inanimate object, such as a rubber duck. The act of rubber duck debugging is simple, yet powerful. The person outlines their current understanding of the problem, as well as the possible solutions. The act of verbalizing the problem can help the person discover new angles of thought. They might identify contradictions or potential loopholes in their logic. By articulating the steps they've already taken, they can ensure that they haven't missed anything important. It is particularly effective for individuals who prefer to work in a more exploratory manner.

# Uploading Dataset to github Problem

__Problem explanation:__ 

2 GB is the limit for files to be uploaded into GitHub, for GitHub free and Pro. However, there may also be limitations on internet connection or the time it takes to upload that large file.

__Problem Solving Solution:__

upload the download links for the DataSets instead.

# Different image size and different image file type

__Problem explanation:__ 

in the datasets there are different files from .jpg, .png, .svg there is a need for normalizing all the data with the size and name.

__Problem Solving Solution:__

created a python file to preproccess and rename all the file into a structured way this file is avilable with [utilities.py](./vars/utilities.py) name.

# Calling  modules from other folder

__Problem explanation:__ 

cannot call modules from vars folder into a testing file or workspapce jupyternotebook

__Problem Solving Solution:__

creating `__init__` file in every folder initaties that the files contains modules like library and by this we can call the modules like 
```python
import vars.loading_data
from vars.loading_data import ChessDB as DB
```
# Testing some files that don't return anything

__Problem explanation:__ 

some functions dosent return anything or just plot pictures and graphs,

__Problem Solving Solution:__

we can simply run them and put:
```python
self.assert_("Runs Successfully")
```
this will notice us if the function run sussesfully without errrors.

# Cannot load 2 different ways of loading at the same time

__Problem explanation:__ 

cannot load fully load and weight only load at the same time for the model 

__Problem Solving Solution:__

no solution yet.




