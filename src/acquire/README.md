# Wellscanner: In situ 3D-analysis of aquatic microinvertebrates

### Setting up the wellscanner environment

```
# Install all conda components simultaneously
conda create -n wellscan24_env -c defaults -c conda-forge python=3.9 pip numpy pandas matplotlib opencv scipy

# Install Pylon
cd pypylon
python setup.py install
cd ..

# Install Optotune Python SDK
python -m pip install --no-user C:/Seafile/FWF_PROJEKT/1_mavirops/3_liquid_lens/optotune_python_sdk/ICC-4C_PythonSDK_2.0.4922/optoKummenberg-1.0.4894-py3-none-any.whl
python -m pip install --no-user C:/Seafile/FWF_PROJEKT/1_mavirops/3_liquid_lens/optotune_python_sdk/ICC-4C_PythonSDK_2.0.4922/optoICC-2.0.4922-py3-none-any.whl

# Install deprecated Python library to Control XY table
python -m pip install zaber.serial

# Export Environment (platform-independent)
conda wellscan24_env export --from-history > wellscan24_env.yml

#
conda install anaconda::scikit-image
conda install conda-forge::scikit-learn
python -m pip install pyyaml
python -m pip install pillow # For Image augmentation

python -m pip install psutil

```
