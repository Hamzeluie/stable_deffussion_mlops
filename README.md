# Description
this project is an example of using 'cookiecutter' , 'DVC' and 'git' in a real project.in this repository we use stable deffussion to generate defect.

# instruction
* install cookiecutter on local device
* create a repository in github and connect with cookiecutter
* DVC get start
* DVC data versioning
* DVC data pipeline

## install cookiecutter on local device
do this steps one by one
install cookiecutter on env

        pip install cookiecutter

install cookiecutter

        cd ~
        cookiecutter cookiecutter-pypackage/

fill the 

* "full_name": you name
* "email": your email address
* "github_user_name": fill user name of your github account
* "project_name": name of the project 
* "project_slug": name of then project if wants to install with 'pip install <project_slug>'
* "project_short_description": describe your project.
* "pypi_username": user name of pypi
* "version": version of project
* "use_pytest": [y/n]
* "use_black": [y/n]
* "use_pypi_deployment_with_travis": [y/n]
* "add_pyup_badge": [y/n]
* "Select command_line_interface": [1/2/3]
* "create_author_file": [y/n]
* "Select open_source_license": [1/2/3/4/5/6]

## create a repository in github and connect with cookiecutter
After installing cookiecutter 
go to github and create new repository with name of 'project_slug'
then follow the steps to connect cookiecutter project to github repository.

        cd <project_slug>
        git init
        git add .
        git commit -m "first commit"
        git branch -M main
        git remote add origin https://github.com/<github_username>/<project_slug>.git
        git push -u origin main
## DVC get start
initialize DVC and commit to github

        dvc init
        git commit -m "Initialize DVC"

**directory structure**

        .
        ├── AUTHORS.rst
        ├── CONTRIBUTING.rst
        ├── docs
        │   ├── authors.rst
        │   ├── conf.py
        │   ├── contributing.rst
        │   ├── history.rst
        │   ├── index.rst
        │   ├── installation.rst
        │   ├── make.bat
        │   ├── Makefile
        │   ├── readme.rst
        │   └── usage.rst
        ├── HISTORY.rst
        ├── LICENSE
        ├── Makefile
        ├── MANIFEST.in
        ├── <project_slug>
        │   ├── cli.py
        │   ├── __init__.py
        │   └── <project_slug>.py
        ├── README.rst
        ├── requirements_dev.txt
        ├── setup.cfg
        ├── setup.py
        ├── tests
        │   ├── __init__.py
        │   └── test_<project_slug>.py
        └── tox.ini

make new directory with name of data to put your data on it

        mkdir data

rename directory of <project_slug> to model to put your model files in it

                mv /path/to/<project_slug>/<project_slug> /path/to/<project_slug>/models

make new directory results to outputs in this directory

        mkdir results

make newe directory of src to put train.py , inference.py , ...

        mkdir src

**note:** *data* , *models* and *results* contain files which are being stored and versioned by DVC and *src* contains scripts for training and evaluating the model as well as tests and scripts for pipelines and APIs.

**note:** you can make new directory *notebooks* to Jupyter Notebooks used for the exploratory analysis, development of models, or data manipulation.

Then do

        git add .
        git commit -m "dvc data structure updated"
        git push
# DVC data versioning
now its time to put your data in data directory.
after that you should 'dvc add'
        
        dvc add data/images

after running the up command. **images.dvc** and **.gitignore** file will create in data directory.
in gitignore wrote **/images** to ignore uploading this directiory to github beacuse it should upload to dvc.

**challeng:** in adding a directory contain images to "dvc add data/images" if dvc got  permission error do this

        chmod +rwx path/to/images

**note:** images directory contain multi images should add like this *dvc add data/images*  not this *dvc add data/images/*

**note:** images.dvc contain information of your images in .dvc/cache directory.take care of it. if it remove after pushing dvc . you can not pull your data again. 

its time to git add

        git add data/images.dvc data/.gitignore
        git commit -m "Add raw data"

now its time to add dvc remote to googleDrive but before that you should install its package 

        pip install dvc[gdrive]

so add remote to your google drive
lets imagine its your googleDrive folder path

**https://drive.google.com/drive/folders/14BN_D1HWKFw0POOJHPf5HqjW1bZayNCC**

put last part of path in remote add

        dvc remote add -d <remote_name> gdrive://14BN_D1HWKFw0POOJHPf5HqjW1bZayNCC

now dvc push to upload data to googleDrive

        dvc push