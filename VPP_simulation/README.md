# PRV_simulation

This folder contains the view plannning simulation system for active NeRF reconstruction.

## Installion

These libraries need to be installed: opencv 4.4.0, PCL 1.9.1, Eigen 3.3.9, OctoMap 1.9.6, Gurobi 10.0.0 (free for academic use), and JsonCpp.

Our codes can be compiled by Visual Studio 2022 with c++ 14 and run on Windows 11.

For other system, please check the file read/write or multithreading functions in the codes.

## Perpare

### A. Test 3D Models

1. Download ["HomebrewedDB"]](https://campar.in.tum.de/personal/ilic/homebreweddb/index.html) dataset and put *.ply to 3D_models/PLY folder.

### B. View Spaces

Use our processed view spaces in Hemisphere subfolder.
Or you want to generate them:

1. Download Tammes_sphere from [Tammes-problem](https://github.com/XiangjingLai/Tammes-problem).
2. Change orginalviews_path in DefaultConfiguration.yaml.
3. Run with mode = 0 (GetViewSpace) with -1.

## Main Usage

DefaultConfiguration.yaml contains most parameters.

The mode of the system should be input in the Console. These modes are for different functions as follows.

Then give the object model names in the Console (-1 to break input).

## View Planning Comparsion

1. Run with mode = 1 (ViewPlanning). Note the first-time running will generate gournd truth images, which takes some time.
2. Input object names that you want to test and -1.
3. For ["One-2-3-45++"](https://sudo-ai-3d.github.io/One2345plus_page/), Follow the instructions in the Console to obatin the generated mesh, as they have only demo.
4. Check the folders in the repository: instantngp_scripts, Mesh_scripts, PRVNet_scripts for supporting functions.

Change use_gt_mesh to 1 in DefaultConfiguration.yaml to enable gournd truth covering ablation.  
Change 2138-2150 lines in main.cpp for different methods and init_views (default run our method).

