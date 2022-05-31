# imageJ_wholeBrainAtlas
Package for ImageJ to have analysis output of RoiManagerX which is equivalent to ImageJ's native ROI manager with added functionality for brain mapping. And downstream analysis of the data

## TODO
	* work on documentation of README
	* work on aggregation documentation


## Protocols
The protocol folder contains series of protocol initially used for quantfication and registration of cells using imageJ. The prefered method is now to use [wholebrain](https://github.com/tractatus/wholebrain)

## ImageJ RoiManager X
	* Java and ImageJ need to be installed for the pluglin to work
	* The extracted plugin can then be copied into the ImageJ/Fiji plugin folder
	* Most of the development is located in the subfolder `edu/scripps` within the RoiManager X

## Downstream aggregation
	* `forPostProcessing.py`enables file aggregation of exported output