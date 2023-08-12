import os
from uuid import uuid4
#from demian.views import new_name

rename=[]
def rename_file_to_uuid(instance, filename):
        upload_to = "textuploads/"
        ext = filename.split('.')[-1]
        uuid = uuid4().hex
        
        filename = '{}.{}'.format(uuid, ext)
        rename.append(filename)
        
        return os.path.join(upload_to, filename)