import numpy as np
import torch

baseURL = '/home/l_z80934/projects/zip/damon/train/'

# data_contact = np.load("contact_label_smplx.npy", allow_pickle=True)

# print(type(data))
# print(data.shape)   # (4384, 10475)
# print(data.dtype)
# np.savetxt("data_0.txt", data_contact[0])


# img = np.load(baseURL+"imgname.npy", allow_pickle=True)     # (4384)
# np.savetxt("img_0.txt", img[0])
# print(img[0])



# objects = np.load(baseURL+"contact_label_smplx_objectwise.npy", allow_pickle=True)
# np.savetxt("object_0.txt", objects[0])
# print(objects[0])


# objects = np.load(baseURL+"contact_label_smplx.npy", allow_pickle=True)     # smplx vertices: 10475     (4384,10475)
# np.savetxt("contact_label_smplx_0.txt", objects[0])
# print(objects[0])

# objects = np.load(baseURL+"body_parts_all.pkl", allow_pickle=True)      # dict: (4375)   # smplx vertices: 10475     (4384,10475)
# np.savetxt("body_parts_all_0.txt", objects[0])
# print(objects[0])

# objects = np.load(baseURL+"body_parts_objectwise.pkl", allow_pickle=True)      # dict: (7811)
# np.savetxt("body_parts_objectwise_0.txt", objects[0])
# print(objects[0])

# objects = np.load(baseURL+"body_parts_objectwise_wFootGround.pkl", allow_pickle=True)       # (10437)
# np.savetxt("body_parts_objectwise_wFootGround_0.txt", objects[0])
# print(objects[0])

# objects = np.load(baseURL+"contact_label_wFootGround.npy", allow_pickle=True)       # (4384,)
# np.savetxt("contact_label_wFootGround_0.txt", objects[0])
# print(objects[0])


# objects = np.load(baseURL+"part_seg.npy", allow_pickle=True)       # (4384,)
# np.savetxt("part_seg_0.txt", objects[0])
# print(objects[0])

# objects = np.load(baseURL+"polygon_2d_contact.npy", allow_pickle=True)       # (4384,)
# np.savetxt("polygon_2d_contact_0.txt", objects[0])
# print(objects[0])

# objects = np.load(baseURL+"pose.npy", allow_pickle=True)       # (4384, 72)
# np.savetxt("pose_0.txt", objects[0])
# print(objects[0])

# objects = np.load(baseURL+"scene_seg.npy", allow_pickle=True)       # (4384,)
# np.savetxt("scene_seg_0.txt", objects[0])
# print(objects[0])

# objects = np.load(baseURL+"shape.npy", allow_pickle=True)       # (4384, 10)
# np.savetxt("shape_0.txt", objects[0])
# print(objects[0])


# objects = np.load(baseURL+"transl.npy", allow_pickle=True)       # (4384, 3)
# np.savetxt("transl_0.txt", objects[0])
# print(objects[0])


# objects = np.load(baseURL+"contact_label_objectwise.npy", allow_pickle=True)       # (4384, )
# np.savetxt("contact_label_objectwise_0.txt", objects[0])
# print(objects[0])


# objects = np.load(baseURL+"contact_label.npy", allow_pickle=True)       # (4384, 6890)
# np.savetxt("contact_label_0.txt", objects[0])
# print(objects[0])


# objects = np.load(baseURL+"contact_label_objectwise_wFootGround.pkl", allow_pickle=True)       # (4384,)
# np.savetxt("contact_label_objectwise_wFootGround_0.txt", objects[0])
# print(objects[0])

# objects = np.load(baseURL+"cam_k.npy", allow_pickle=True)       # (4384, 3, 3)
# np.savetxt("cam_k_0.txt", objects[0])
# print(objects[0])

objects = torch.load("/home/l_z80934/projects/sam-3d-body/datasets/damon/batch_0.pt", map_location="cpu", weights_only=False)       # np.load("./datasets/damon/batch_0.pt", allow_pickle=True)
print(objects[0])

