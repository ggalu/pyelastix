import itk
import numpy as np
import os, sys
from StrainComputer import StrainComputer
import tifffile

output_directory = "fgs10"
output_filetype = "tif"
num_resolutions = 5
final_grid_spacing = 10.0
num_iterations = 512 #
num_samples = 2048
bending_penalty = 1.0e2 # works at bin4 with FGS 10
internal_datatype = "short"
initial_transform = "affine/TransformParameters.1.txt" # typically set to an initial rigid-affine registration


# Import Images

def image_importer(filename):
    im = tifffile.imread(filename).astype(np.float32) / (2**16 - 1) # read and rescale to range [0,1]
    im = im * (2**14 - 1) # scale to 14 bit range
    im = im.astype(np.int16) # cast to short, suitable for elastix
    return itk.image_view_from_array(im)


fixed_image = image_importer('data/D2_cropped.tif')
moving_image = image_importer('data/D6_cropped.tif')

fixed_mask = np.zeros(fixed_image.shape, np.uint8) # z-dimension (vertical) is first
fixed_mask[:] = 1
fixed_mask = itk.image_view_from_array(fixed_mask)


# define b-spline registration
parameter_object = itk.ParameterObject.New()
parameter_map_bspline = parameter_object.GetDefaultParameterMap("bspline", num_resolutions, final_grid_spacing)
parameter_object.AddParameterMap(parameter_map_bspline)
parameter_object.SetParameter("MaximumNumberOfIterations", (f"{num_iterations}",))
parameter_object.SetParameter("NumberOfSpatialSamples", (f"{num_samples}",))
parameter_object.SetParameter("ResultImageFormat", output_filetype)
parameter_object.SetParameter("Metric", ("AdvancedMattesMutualInformation", "TransformBendingEnergyPenalty"))
parameter_object.SetParameter("Metric0Weight", (f"{1}",))
parameter_object.SetParameter("Metric1Weight", (f"{bending_penalty}",))
parameter_object.SetParameter("FixedInternalImagePixelType", internal_datatype) # short is also supported, but need to cast input images to that type first
parameter_object.SetParameter("MovingInternalImagePixelType", internal_datatype)
##parameter_object.SetParameter("ShowExactMetricValue", ("true",)) # infinitely slow


print(parameter_object)
# Call registration function
if True:

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)


    result_image, result_transform_parameters = itk.elastix_registration_method(
        fixed_image, moving_image,
        parameter_object=parameter_object,
        initial_transform_parameter_file_name=initial_transform,
        log_to_console=True,
        output_directory=output_directory,
        fixed_mask=fixed_mask)
    
# compute displacements and strain with transformix
if True:
    print("... saving displacements and deformation gradient")
    result_transform_parameters = itk.ParameterObject.New()
    result_transform_parameters.ReadParameterFile(output_directory+"/TransformParameters.0.txt")

    transformix_object = itk.TransformixFilter.New(moving_image)
    transformix_object.SetTransformParameterObject(result_transform_parameters)

    transformix_object.SetComputeDeformationField(True)
    transformix_object.SetComputeSpatialJacobian(True)
    transformix_object.SetOutputDirectory(output_directory)
    transformix_object.SetLogToConsole=True,
    transformix_object.UpdateLargestPossibleRegion() # Update object (required)

    # Results of Transformation
    result_image_transformix = transformix_object.GetOutput()
    deformation_field = transformix_object.GetOutputDeformationField()

# compute stretch tensor diagonals (Exx, Eyy, Ezz) from deformation gradient
if True:
    strainComputer = StrainComputer(output_directory+"/fullSpatialJacobian.tif", margin=0)

