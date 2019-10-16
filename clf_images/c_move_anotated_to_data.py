import os
from xml.dom import minidom

path_anotations = './clf_images/anotated_data'
out_folder = './clf_images/data/images'
out_folder_xml = './clf_images/data/xmls'
os.makedirs(out_folder,exist_ok=True)
os.makedirs(out_folder_xml,exist_ok=True)



# Lee carpeta 
a_files = os.listdir(path_anotations)

# lee <path>, lista de objetos
for xml_f in a_files:
    print(xml_f)
    tipo = xml_f.split('.')[-1]
    path_full = os.path.join(path_anotations,xml_f)
    if tipo == 'xml':
        # lee archivos xml
        x=minidom.parse(path_full)
        assert(len(x.getElementsByTagName('path')) > 0)
        
        path_node = x.getElementsByTagName('path')[0].childNodes[0] 
        path_img = path_node.data

        assert(os.path.exists(path_img)),'No existe imagen {0}'.format(path_img)

        # mueve imagen referenciada en xml
        img_name = os.path.split(path_img)[1]
        new_path = os.path.join(out_folder,img_name)
        os.rename(path_img,new_path)

        # edita xml
        path_node.data = new_path


        # escribe xml modificado
        out_path_xml = os.path.join(out_folder_xml,xml_f)
        with open(out_path_xml,'w') as f:
            f.write(x.toxml())
        
        # borra antiguo xml
        os.remove(path_full)


