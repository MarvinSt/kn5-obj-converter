import datetime
import math
import struct
import os
import numpy as np


class kn5Material:
    def __init__(self):
        self.name = ""
        self.shader = ""
        self.ksAmbient = 0.6
        self.ksDiffuse = 0.6
        self.ksSpecular = 0.9
        self.ksSpecularEXP = 1.0
        self.diffuseMult = 1.0
        self.normalMult = 1.0
        self.useDetail = 0.0
        self.detailUVMultiplier = 1.0
        self.txDiffuse = ""
        self.txNormal = ""
        self.txDetail = ""
        self.shader_props = ""


class kn5Node:
    def __init__(self):
        self.name = "Default"
        self.parent = None
        self.tmatrix = np.identity(4)
        self.hmatrix = np.identity(4)
        # self.children = []
        self.meshIndex = -1
        self.type = 1
        self.materialID = -1

        self.translation = np.identity(3)
        self.rotation = np.identity(3)
        self.scaling = np.identity(3)

        self.vertexCount = 0
        self.indices = []

        self.position = []
        self.normal = []
        self.texture0 = []


def read_string(file, length):
    # Read a UTF-8-encoded string from the file
    string_data = file.read(length)
    string = string_data.decode("utf-8")
    return string


def matrix_mult(ma, mb):
    mm = [[0.0 for j in range(4)] for i in range(4)]

    for i in range(4):
        for j in range(4):
            mm[i][j] = ma[i][0] * mb[0][j] + ma[i][1] * \
                mb[1][j] + ma[i][2] * mb[2][j] + ma[i][3] * mb[3][j]

    return mm


def matrix_to_euler(transf):
    heading = 0
    attitude = 0
    bank = 0

    # original code by Martin John Baker for right-handed coordinate system
    """
    if (transf[0, 1] > 0.998):
        # singularity at north pole
        heading = math.atan2(transf[0, 2], transf[2, 2])
        attitude = math.pi / 2
        bank = 0
    
    if (transf[0, 1] < -0.998):
        # singularity at south pole
        heading = math.atan2(transf[0, 2], transf[2, 2])
        attitude = -math.pi / 2
        bank = 0
    """

    # left-handed
    if (transf[0][1] > 0.998):
        # singularity at north pole
        heading = np.arctan2(-transf[1][0], transf[1][1])
        attitude = -math.pi / 2
        bank = 0.0
    elif (transf[0][1] < -0.998):
        # singularity at south pole
        heading = np.arctan2(-transf[1][0], transf[1][1])
        attitude = math.pi / 2
        bank = 0.0
    else:
        heading = np.arctan2(transf[0][1], transf[0][0])
        bank = np.arctan2(transf[1][2], transf[2][2])
        attitude = np.arcsin(-transf[0][2])

    # alternative code by Mike Day, Insomniac Games
    """
    bank = math.atan2(transf[1, 2], transf[2, 2])
    c2 = math.sqrt(transf[0, 0] * transf[0, 0] + transf[0, 1] * transf[0, 1])
    attitude = math.atan2(-transf[0, 2], c2)
    s1 = math.sin(bank)
    c1 = math.cos(bank)
    heading = math.atan2(s1 * transf[2, 0] - c1 * transf[1, 0], c1 * transf[1, 1] - s1 * transf[2, 1])
    """

    attitude *= 180 / math.pi
    heading *= 180 / math.pi
    bank *= 180 / math.pi

    return [bank, attitude, heading]


def scale_from_matrix(transf):
    scaleX = math.sqrt(transf[0][0] * transf[0][0] + transf[1][0]
                       * transf[1][0] + transf[2][0] * transf[2][0])
    scaleY = math.sqrt(transf[0][1] * transf[0][1] + transf[1][1]
                       * transf[1][1] + transf[2][1] * transf[2][1])
    scaleZ = math.sqrt(transf[0][2] * transf[0][2] + transf[1][2]
                       * transf[1][2] + transf[2][2] * transf[2][2])
    return [float(scaleX), float(scaleY), float(scaleZ)]


def read_nodes(file, node_list, parent_id):
    new_node = kn5Node()
    new_node.parent = parent_id

    new_node.type, = struct.unpack('<i', file.read(4))
    new_node.name = read_string(file, struct.unpack('<i', file.read(4))[0])
    children_count, = struct.unpack('<i', file.read(4))
    abyte = file.read(1)

    if new_node.type == 1:  # dummy
        new_node.tmatrix = [[0] * 4 for _ in range(4)]
        for i in range(4):
            for j in range(4):
                new_node.tmatrix[i][j], = struct.unpack('<f', file.read(4))

        new_node.translation = [new_node.tmatrix[3][0],
                                new_node.tmatrix[3][1],
                                new_node.tmatrix[3][2]]

        new_node.rotation = matrix_to_euler(new_node.tmatrix)
        new_node.scaling = scale_from_matrix(new_node.tmatrix)

    elif new_node.type == 2:  # mesh
        bbyte = file.read(1)
        cbyte = file.read(1)
        dbyte = file.read(1)

        new_node.vertexCount, = struct.unpack('<i', file.read(4))
        new_node.position = []
        new_node.normal = []
        new_node.texture0 = []

        for v in range(new_node.vertexCount):
            new_node.position.extend(struct.unpack('<fff', file.read(12)))
            new_node.normal.extend(struct.unpack('<fff', file.read(12)))
            new_node.texture0.extend(struct.unpack('<ff', file.read(8)))
            # file.seek(12, 1)  # tangents
            file.read(12)

        index_count, = struct.unpack('<i', file.read(4))
        new_node.indices = struct.unpack(
            '<%dH' % index_count, file.read(index_count * 2))
        new_node.materialID, = struct.unpack('<i', file.read(4))
        # file.seek(29, 1)
        file.read(29)

    elif new_node.type == 3:  # animated mesh
        bbyte = file.read(1)
        cbyte = file.read(1)
        dbyte = file.read(1)

        bone_count, = struct.unpack('<i', file.read(4))
        for b in range(bone_count):
            bone_name = read_string(file, struct.unpack('<i', file.read(4))[0])
            # file.seek(64, 1)  # transformation matrix
            file.read(64)

        new_node.vertexCount, = struct.unpack('<i', file.read(4))
        new_node.position = []
        new_node.normal = []
        new_node.texture0 = []
        for v in range(new_node.vertexCount):
            new_node.position.extend(struct.unpack('<fff', file.read(12)))
            new_node.normal.extend(struct.unpack('<fff', file.read(12)))
            new_node.texture0.extend(struct.unpack('<ff', file.read(8)))
            # file.seek(44, 1)  # tangents & weights
            file.read(44)

        index_count, = struct.unpack('<i', file.read(4))
        new_node.indices = struct.unpack(
            '<%dH' % index_count, file.read(index_count * 2))
        new_node.materialID, = struct.unpack('<i', file.read(4))
        # file.seek(12, 1)
        file.read(12)

    if parent_id < 0:
        new_node.hmatrix = new_node.tmatrix
    else:
        new_node.hmatrix = matrix_mult(
            new_node.tmatrix, node_list[parent_id].hmatrix)

    node_list.append(new_node)
    current_id = len(node_list) - 1

    for c in range(children_count):
        node_list = read_nodes(file, node_list, current_id)

    return node_list


def export_obj(model_name, output_dir, materials: list[kn5Material], nodes: list[kn5Node]):
    model_filename = model_name

    # if os.path.exists(os.path.join(output_dir, model_filename + '.obj')):
    #     return

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    skip_ac_nodes = True

    print('Exporting {}.obj'.format(model_filename))

    # write MTL
    with open(os.path.join(output_dir, model_filename + '.mtl'), 'w') as mtl_writer:
        mtl_str = ''

        for src_mat in materials:
            mtl_str += 'newmtl {}\r\n'.format(
                src_mat.name.replace(' ', '_'))
            mtl_str += 'Ka {} {} {}\r\n'.format(
                src_mat.ksAmbient, src_mat.ksAmbient, src_mat.ksAmbient)
            mtl_str += 'Kd {} {} {}\r\n'.format(
                src_mat.ksDiffuse, src_mat.ksDiffuse, src_mat.ksDiffuse)
            mtl_str += 'Ks {} {} {}\r\n'.format(
                src_mat.ksSpecular, src_mat.ksSpecular, src_mat.ksSpecular)
            mtl_str += 'Ns {}\r\n'.format(src_mat.ksSpecularEXP)
            mtl_str += 'illum 2\r\n'
            if src_mat.useDetail == 1.0 and src_mat.txDetail:
                mtl_str += 'map_Kd texture\\{}\r\n'.format(
                    src_mat.txDetail)
                mtl_str += 'map_d texture\\{}\r\n'.format(
                    src_mat.txDetail)
                if src_mat.txDiffuse:
                    mtl_str += 'map_Ks texture\\{}\r\n'.format(
                        src_mat.txDiffuse)
            elif src_mat.txDiffuse:
                mtl_str += 'map_Kd texture\\{}\r\n'.format(
                    src_mat.txDiffuse)
                mtl_str += 'map_d texture\\{}\r\n'.format(
                    src_mat.txDiffuse)
            if src_mat.txNormal:
                mtl_str += 'bump texture\\{}\r\n'.format(src_mat.txNormal)
            mtl_str += 'd 0.9999\r\n'
            mtl_str += '\r\n'

        mtl_writer.write(mtl_str)

    # write OBJ
    with open(os.path.join(output_dir, model_filename + ".obj"), "w") as OBJwriter:
        sb = "# Assetto Corsa model\n"
        sb += f"# Exported with kn5 Converter by Chipicao on {datetime.datetime.now()}\n"
        sb += f"\nmtllib {model_filename}.mtl\n"

        vertexPad = 1

        for srcNode in nodes:
            if skip_ac_nodes and srcNode.name.startswith("AC_"):
                continue

            if srcNode.type == 1:
                continue
            elif srcNode.type in [2, 3]:
                sb += f"\n g {srcNode.name.replace(' ', '_')}\n"

                for v in range(srcNode.vertexCount):
                    x = srcNode.position[v * 3]
                    y = srcNode.position[v * 3 + 1]
                    z = srcNode.position[v * 3 + 2]

                    vx = srcNode.hmatrix[0][0] * x + srcNode.hmatrix[1][0] * \
                        y + srcNode.hmatrix[2][0] * \
                        z + srcNode.hmatrix[3][0]
                    vy = srcNode.hmatrix[0][1] * x + srcNode.hmatrix[1][1] * \
                        y + srcNode.hmatrix[2][1] * \
                        z + srcNode.hmatrix[3][1]
                    vz = srcNode.hmatrix[0][2] * x + srcNode.hmatrix[1][2] * \
                        y + srcNode.hmatrix[2][2] * \
                        z + srcNode.hmatrix[3][2]

                    sb += f"v {vx} {vy} {vz}\n"

                OBJwriter.write(sb)
                sb = ""

                for v in range(srcNode.vertexCount):
                    x = srcNode.normal[v * 3]
                    y = srcNode.normal[v * 3 + 1]
                    z = srcNode.normal[v * 3 + 2]

                    nx = srcNode.hmatrix[0][0] * x + \
                        srcNode.hmatrix[1][0] * y + \
                        srcNode.hmatrix[2][0] * z
                    ny = srcNode.hmatrix[0][1] * x + \
                        srcNode.hmatrix[1][1] * y + \
                        srcNode.hmatrix[2][1] * z
                    nz = srcNode.hmatrix[0][2] * x + \
                        srcNode.hmatrix[1][2] * y + \
                        srcNode.hmatrix[2][2] * z

                    sb += f"vn {nx} {ny} {nz}\n"

                OBJwriter.write(sb)
                sb = ""

                UVmult = 1.0
                if srcNode.materialID >= 0:
                    if materials[srcNode.materialID].useDetail == 0.0:
                        UVmult = materials[srcNode.materialID].diffuseMult
                    else:
                        UVmult = materials[srcNode.materialID].detailUVMultiplier

                for v in range(srcNode.vertexCount):
                    tx = srcNode.texture0[v * 2] * UVmult
                    ty = srcNode.texture0[v * 2 + 1] * UVmult

                    sb += f"vt {tx} {ty}\n"

                OBJwriter.write(sb)
                sb = []

                if srcNode.materialID >= 0:
                    sb.append("\r\nusemtl {}\r\n".format(
                        materials[srcNode.materialID].name.replace(' ', '_')))
                else:
                    sb.append("\r\nusemtl Default\r\n")

                for i in range(0, len(srcNode.indices) // 3):
                    i1 = srcNode.indices[i * 3] + vertexPad
                    i2 = srcNode.indices[i * 3 + 1] + vertexPad
                    i3 = srcNode.indices[i * 3 + 2] + vertexPad

                    sb.append(
                        "f {}/{}/{} {}/{}/{} {}/{}/{}\r\n".format(i1, i1, i1, i2, i2, i2, i3, i3, i3))

                sb = "".join(sb)
                OBJwriter.write(sb)

                vertexPad += srcNode.vertexCount
                continue


def read_kn5(file_path, output_dir):

    textures = []
    materials = []
    meshes = []

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Open the file in binary mode and read the header data
    with open(file_path, "rb") as file:
        header = file.read(10)
        # Parse the header data
        magic, version = struct.unpack("<6s1I", header)

        if version > 5:
            _ = file.read(4)

        # Extract textures
        tex_count = struct.unpack("<i", file.read(4))[0]
        for t in range(tex_count):
            tex_type = struct.unpack("<i", file.read(4))[0]
            tex_name = read_string(file, struct.unpack("<i", file.read(4))[0])
            tex_size = struct.unpack("<i", file.read(4))[0]
            textures.append(tex_name)

            tex_path = os.path.join(output_dir, "texture", tex_name)
            if os.path.exists(tex_path):
                file.seek(tex_size, os.SEEK_CUR)
            else:
                if not os.path.exists(os.path.dirname(tex_path)):
                    os.mkdir(os.path.dirname(tex_path))

                with open(tex_path, "wb") as tex_file:
                    tex_file.write(file.read(tex_size))

        # Extract materials
        mat_count = struct.unpack("<i", file.read(4))[0]

        for m in range(mat_count):
            new_material = kn5Material()

            new_material.name = read_string(
                file, struct.unpack("<i", file.read(4))[0])
            new_material.shader = read_string(
                file, struct.unpack("<i", file.read(4))[0])
            ashort = struct.unpack("<h", file.read(2))[0]
            if version > 4:
                azero = struct.unpack("<i", file.read(4))[0]

            prop_count = struct.unpack("<i", file.read(4))[0]
            for p in range(prop_count):
                prop_name = read_string(
                    file, struct.unpack("<i", file.read(4))[0])
                prop_value = struct.unpack("<f", file.read(4))[0]
                new_material.shader_props += prop_name + \
                    " = " + str(prop_value) + "&cr;&lf;"

                if prop_name == "ksAmbient":
                    new_material.ksAmbient = prop_value
                elif prop_name == "ksDiffuse":
                    new_material.ksDiffuse = prop_value
                elif prop_name == "ksSpecular":
                    new_material.ksSpecular = prop_value
                elif prop_name == "ksSpecularEXP":
                    new_material.ksSpecularEXP = prop_value
                elif prop_name == "diffuseMult":
                    new_material.diffuseMult = prop_value
                elif prop_name == "normalMult":
                    new_material.normalMult = prop_value
                elif prop_name == "useDetail":
                    new_material.useDetail = prop_value
                elif prop_name == "detailUVMultiplier":
                    new_material.detailUVMultiplier = prop_value

                file.seek(36, os.SEEK_CUR)

            textures = struct.unpack("<i", file.read(4))[0]
            for t in range(textures):
                sample_name = read_string(
                    file, struct.unpack("<i", file.read(4))[0])
                sample_slot = struct.unpack("<i", file.read(4))[0]
                tex_name = read_string(
                    file, struct.unpack("<i", file.read(4))[0])

                new_material.shader_props += sample_name + " = " + tex_name + "&cr;&lf;"

                if sample_name == "txDiffuse":
                    new_material.txDiffuse = tex_name
                elif sample_name == "txNormal":
                    new_material.txNormal = tex_name
                elif sample_name == "txDetail":
                    new_material.txDetail = tex_name

            materials.append(new_material)

        # read meshes
        meshes = read_nodes(file, meshes, -1)

        # material_count, mesh_count = struct.unpack("<6s3I", header)

        # # Read the materials
        # materials = []
        # for _ in range(material_count):
        #     material_header = file.read(16)
        #     name_length, unknown1, unknown2, unknown3 = struct.unpack(
        #         "<4I", material_header)
        #     material_name = read_string(file, name_length)
        #     materials.append(material_name)

        # # Read the meshes
        # meshes = []
        # for _ in range(mesh_count):
        #     mesh_header = file.read(16)
        #     name_length, unknown1, unknown2, mesh_data_size = struct.unpack(
        #         "<4I", mesh_header)
        #     mesh_name = read_string(file, name_length)
        #     mesh_data = file.read(mesh_data_size)
        #     meshes.append((mesh_name, mesh_data))

        # good idea to export here...

        # Return the parsed data
        return textures, materials, meshes


def convert_to_obj(file_path):
    # extract model name from file and generate output folder
    model_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.join(os.path.dirname(file_path), "output")

    _, materials, meshes = read_kn5(file_path, output_dir)
    export_obj(model_name, output_dir, materials, meshes)


if __name__ == "__main__":
    model_name = "my-model.kn5"
    folder_path = f"./models/{model_name}/"
    extension = ".kn5"

    # List all files in the folder with the specified extension
    files = [file for file in os.listdir(
        folder_path) if file.endswith(extension)]

    # Print each file name
    for file in files:
        convert_to_obj(os.path.join(folder_path, file))
