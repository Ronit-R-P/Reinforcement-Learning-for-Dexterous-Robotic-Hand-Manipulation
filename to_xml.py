import mujoco
model = mujoco.MjModel.from_xml_path("mujoco hand.urdf")
mujoco.mj_saveLastXML("HandV4.xml", model)