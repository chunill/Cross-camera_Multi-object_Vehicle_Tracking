import gdown

id = "input your download id"
output_pth = "input your output path"

gdown.download_folder(id=id, output=output_pth, quiet=False)