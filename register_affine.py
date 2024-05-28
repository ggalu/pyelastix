import itk, os

output_directory = "affine"
output_filetype = "tif"

# Import Images
PixelType = itk.ctype("unsigned short") # # uint16 shorthand: itk.UC
fixed_image = itk.imread('data/D2_cropped.tif', PixelType)
moving_image = itk.imread('data/D6_cropped.tif', PixelType)

# Import Default Parameter Map
parameter_object = itk.ParameterObject.New()
num_resolutions = 5
parameter_map_rigid = parameter_object.GetDefaultParameterMap('rigid', num_resolutions )
parameter_object.AddParameterMap(parameter_map_rigid)

parameter_map_affine = parameter_object.GetDefaultParameterMap('affine', num_resolutions )
parameter_object.AddParameterMap(parameter_map_affine)

parameter_object.SetParameter("ResultImageFormat", output_filetype)

print(parameter_object)

# create output directory
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Call registration function
if True:
    result_image, result_transform_parameters = itk.elastix_registration_method(
        fixed_image, moving_image,
        parameter_object=parameter_object,
        log_to_console=True,
        output_directory=output_directory)