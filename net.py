from torch.nn.utils.rnn import *
import articulate as art
from articulate.utils.torch import *
from config import *
from utils import *
from dynamics import PhysicsOptimizer
import torch
import pdb

class PIP(torch.nn.Module):
    name = 'PIP'
    n_hidden = 256

    def __init__(self):
        super(PIP, self).__init__()
        self.simplified = True
        self.is_initialized = False # Determine whether the model is initialized
        self.lj_init = None
        self.jvel_init = None

        # This class is defined by the author,
        # It takes extra input to calculate initial states using a fully-connected layer inside

        # First input dimension is 72, because 6 sensor acceleration, each has 3d. 3 * 6 = 18
        # Then you got 6 sensor with each 3*3 rotation matrix, then you have: 6 * 9 = 54
        # 54 + 18 = 72

        # Here the output is joint position of the leaf
        self.rnn1 = RNNWithInit(input_size=72,
                                output_size=joint_set.n_leaf * 3,
                                hidden_size=self.n_hidden,
                                num_rnn_layer=2,
                                simplified=self.simplified,
                                dropout=0.4)

        # Then the input & output of the above model are used as input here.
        # Output is 24 joint positions, each has 3 dimension
        self.rnn2 = RNN(input_size=72 + joint_set.n_leaf * 3,
                        output_size=joint_set.n_full * 3,
                        hidden_size=self.n_hidden,
                        num_rnn_layer=2,
                        simplified=self.simplified,
                        dropout=0.4)

        # Then the input & output of the above model, used again as input here...
        # Output is the joint rotations, in 6d version. You should (I think) Use pose to get the corresponding
        # rotations
        self.rnn3 = RNN(input_size=72 + joint_set.n_full * 3,
                        output_size=joint_set.n_reduced * 6,
                        hidden_size=self.n_hidden,
                        num_rnn_layer=2,
                        simplified=self.simplified,
                        dropout=0.4)

        # Then, also this model it predicts
        self.rnn4 = RNNWithInit(input_size=72 + joint_set.n_full * 3,
                                output_size=24 * 3,
                                hidden_size=self.n_hidden,
                                num_rnn_layer=2,
                                simplified=self.simplified,
                                dropout=0.4)
        self.rnn5 = RNN(input_size=72 + joint_set.n_full * 3,
                        output_size=2,
                        hidden_size=64,
                        num_rnn_layer=2,
                        simplified=self.simplified,
                        dropout=0.4)

        body_model = art.ParametricModel(paths.smpl_file)
        # From global rotation to local rotation
        self.inverse_kinematics_R = body_model.inverse_kinematics_R
        # From local to global
        self.forward_kinematics = body_model.forward_kinematics
        self.dynamics_optimizer = PhysicsOptimizer(debug=False)
        self.rnn_states = [None for _ in range(5)]
        # Load the weights automatically when you create this instance!
        self.load_state_dict(torch.load(paths.weights_file))
        self.eval()

    # From global 6d rotation to local rotation
    def _reduced_glb_6d_to_full_local_mat(self, root_rotation, glb_reduced_pose):
        glb_reduced_pose = art.math.r6d_to_rotation_matrix(glb_reduced_pose).view(-1, joint_set.n_reduced, 3, 3)
        global_full_pose = torch.eye(3, device=glb_reduced_pose.device).repeat(glb_reduced_pose.shape[0], 24, 1, 1)
        global_full_pose[:, joint_set.reduced] = glb_reduced_pose
        pose = self.inverse_kinematics_R(global_full_pose).view(-1, 24, 3, 3)
        pose[:, joint_set.ignored] = torch.eye(3, device=pose.device)
        pose[:, 0] = root_rotation.view(-1, 3, 3)
        return pose

    def step_forward(self, x):
        x, lj_init, jvel_init = list(x[0])

        leaf_joint_list = []
        full_joint_list = []
        global_6d_pose_list = []
        joint_velocity_list = []
        contact_list = []
        # Iterate over each frame
        for i, frame in enumerate(x):
            # frame_reshaped = frame.unsqueeze(0).unsqueeze(0)  # Now the shape is [1, 1, 72]
            leaf_joint = self.rnn1(list([frame, lj_init]))
            leaf_joint_list.append(leaf_joint)

            full_joint = self.rnn2(torch.cat((leaf_joint, frame), dim=0))
            full_joint_list.append(full_joint)

            global_6d_pose = self.rnn3(torch.cat((full_joint, frame), dim=0))
            global_6d_pose_list.append(global_6d_pose)
            # Linear velocities
            joint_velocity = self.rnn4(list([torch.cat((full_joint, frame), dim=0), jvel_init]))
            joint_velocity_list.append(joint_velocity)
            # Foot ground contact probability
            contact = self.rnn5(torch.cat((full_joint, frame), dim=0))
            contact_list.append(contact)

        leaf_joint = torch.stack(leaf_joint_list)
        full_joint = torch.stack(full_joint_list)
        global_6d_pose = torch.stack(global_6d_pose_list)
        joint_velocity = torch.stack(joint_velocity_list)
        contact = torch.stack(contact_list)
        return [leaf_joint], [full_joint], [global_6d_pose], [joint_velocity], [contact]

    def forward(self, x):
        r"""
        Forward.

        :param x: A list in length [batch_size] which contains 3-tuple
                  (tensor [num_frames, 72], tensor [15], tensor [72]).
        """
        if self.simplified:
            return self.step_forward(x)
        else:
            # Extract three list from x using zip(*x)
            x, lj_init, jvel_init = list(zip(*x))
            # All leaf positions
            # lj_init is used by the fully-connected network inside to calculate initial states
            leaf_joint = self.rnn1(list(zip(x, lj_init)))
            # All joint positions
            full_joint = self.rnn2([torch.cat(_, dim=-1) for _ in zip(leaf_joint, x)])
            # Global rotations
            global_6d_pose = self.rnn3([torch.cat(_, dim=-1) for _ in zip(full_joint, x)])
            # Linear velocities
            joint_velocity = self.rnn4(list(zip([torch.cat(_, dim=-1) for _ in zip(full_joint, x)], jvel_init)))
            # Foot ground contact probability
            contact = self.rnn5([torch.cat(_, dim=-1) for _ in zip(full_joint, x)])
            return leaf_joint, full_joint, global_6d_pose, joint_velocity, contact

    @torch.no_grad()
    def predict(self, glb_acc, glb_rot, init_pose):
        r"""
        Predict the results for evaluation.

        :param glb_acc: A tensor that can reshape to [num_frames, 6, 3].
        6 here means 5 leaf + 1 root
        :param glb_rot: A tensor that can reshape to [num_frames, 6, 3, 3].
        Rotation matrix, in 6d format
        :param init_pose: A tensor that can reshape to [1, 24, 3, 3].
        :return: Pose tensor in shape [num_frames, 24, 3, 3] and
                 translation tensor in shape [num_frames, 3].
        """
        self.dynamics_optimizer.reset_states()
        init_pose = init_pose.view(1, 24, 3, 3)
        init_pose[0, 0] = torch.eye(3)
        # self.forward_kinematics(init_pose)[1], this returns the joint position
        # Then we select the leaf points at index 0 (actually the first dim is always 1, as we can see in init_pose)
        lj_init = self.forward_kinematics(init_pose)[1][0, joint_set.leaf].view(-1)
        # Seems that for velocity initialization, you got all zeros as input...
        jvel_init = torch.zeros(24 * 3)
        # Prepare input by process the data
        x = (normalize_and_concat(glb_acc, glb_rot), lj_init, jvel_init)
        leaf_joint, full_joint, global_6d_pose, joint_velocity, contact = [_[0] for _ in self.forward([x])]
        # Question, why [0] here? Why you got the first element here?
        # It's just an extra dimension
        # What's the shape of glb_rot
        # glb_rot.view(-1, 6, 3, 3)[:, -1] means you get the last root rotation matrix from 6. It's second dimension!
        pose = self._reduced_glb_6d_to_full_local_mat(glb_rot.view(-1, 6, 3, 3)[:, -1], global_6d_pose)
        joint_velocity = joint_velocity.view(-1, 24, 3).bmm(glb_rot[:, -1].transpose(1, 2)) * vel_scale
        pose_opt, tran_opt = [], []
        for p, v, c, a in zip(pose, joint_velocity, contact, glb_acc):
            p, t = self.dynamics_optimizer.optimize_frame(p, v, c, a)
            pose_opt.append(p)
            tran_opt.append(t)
        pose_opt, tran_opt = torch.stack(pose_opt), torch.stack(tran_opt)
        return pose_opt, tran_opt

    @torch.no_grad()
    def single_step_predict(self, glb_acc, glb_rot, init_pose):
        r"""
        Predict the results for evaluation, but we do the processing in a single step way.

        :param glb_acc: A tensor that can reshape to [num_frames, 6, 3].
        6 here means 5 leaf + 1 root
        :param glb_rot: A tensor that can reshape to [num_frames, 6, 3, 3].
        Rotation matrix, in 6d format
        :param init_pose: A tensor that can reshape to [1, 24, 3, 3].
        :return: Pose tensor in shape [num_frames, 24, 3, 3] and
                 translation tensor in shape [num_frames, 3].
        """
        self.dynamics_optimizer.reset_states()
        init_pose = init_pose.view(1, 24, 3, 3)
        init_pose[0, 0] = torch.eye(3)
        # self.forward_kinematics(init_pose)[1], this returns the joint position
        # Then we select the leaf points at index 0 (actually the first dim is always 1, as we can see in init_pose)
        lj_init = self.forward_kinematics(init_pose)[1][0, joint_set.leaf].view(-1)
        # Seems that for velocity initialization, you got all zeros as input...
        jvel_init = torch.zeros(24 * 3)
        # Iteration
        pose_opt, tran_opt = [], []
        # Iterate over each frame
        for i in range(len(glb_acc)):
            acc = glb_acc[i].unsqueeze(0)
            rot = glb_rot[i].unsqueeze(0)
            frame = normalize_and_concat(acc, rot).squeeze()
            leaf_joint = self.rnn1(list([frame, lj_init]))
            full_joint = self.rnn2(torch.cat((leaf_joint, frame), dim=0))
            global_6d_pose = self.rnn3(torch.cat((full_joint, frame), dim=0))
            # Linear velocities
            joint_velocity = self.rnn4(list([torch.cat((full_joint, frame), dim=0), jvel_init]))
            # Foot ground contact probability
            contact = self.rnn5(torch.cat((full_joint, frame), dim=0))
            # What's the shape of glb_rot
            # glb_rot.view(-1, 6, 3, 3)[:, -1] means you get the last root rotation matrix from 6. It's second dimension!
            pose = self._reduced_glb_6d_to_full_local_mat(rot.view(-1, 6, 3, 3)[:, -1], global_6d_pose)
            # Notice: removed vel_scale
            joint_velocity = joint_velocity.view(-1, 24, 3).bmm(rot[:, -1].transpose(1, 2)) * 1
            step_p, step_t = self.dynamics_optimizer.optimize_frame(pose.squeeze(), joint_velocity.squeeze(), contact.cpu(), acc.cpu().squeeze())
            pose_opt.append(step_p)
            tran_opt.append(step_t)
        pose_opt, tran_opt = torch.stack(pose_opt), torch.stack(tran_opt)
        return pose_opt, tran_opt

    @torch.no_grad()
    def new_data_available(self, glb_acc, glb_rot, init_pose):

        if not self.is_initialized:
            self.is_initialized = True
            self.dynamics_optimizer.reset_states()
            init_pose = init_pose.view(1, 24, 3, 3)
            init_pose[0, 0] = torch.eye(3)
            # self.forward_kinematics(init_pose)[1], this returns the joint position
            # Then we select the leaf points at index 0 (actually the first dim is always 1, as we can see in init_pose)
            self.lj_init = self.forward_kinematics(init_pose)[1][0, joint_set.leaf].view(-1)
            # Seems that for velocity initialization, you got all zeros as input...
            self.jvel_init = torch.zeros(24 * 3)

        # acc = glb_acc.unsqueeze(0)
        # rot = glb_rot.unsqueeze(0)
        acc = glb_acc
        rot = glb_rot
        frame = normalize_and_concat(acc, rot).squeeze()
        leaf_joint = self.rnn1(list([frame, self.lj_init]))
        full_joint = self.rnn2(torch.cat((leaf_joint, frame), dim=0))
        global_6d_pose = self.rnn3(torch.cat((full_joint, frame), dim=0))
        # Linear velocities
        joint_velocity = self.rnn4(list([torch.cat((full_joint, frame), dim=0), self.jvel_init]))
        # Foot ground contact probability
        contact = self.rnn5(torch.cat((full_joint, frame), dim=0))
        # What's the shape of glb_rot
        # glb_rot.view(-1, 6, 3, 3)[:, -1] means you get the last root rotation matrix from 6. It's second dimension!
        pose = self._reduced_glb_6d_to_full_local_mat(rot.view(-1, 6, 3, 3)[:, -1], global_6d_pose)
        # Notice: removed vel_scale
        joint_velocity = joint_velocity.view(-1, 24, 3).bmm(rot[:, -1].transpose(1, 2)) * 1
        step_p, step_t = self.dynamics_optimizer.optimize_frame(pose.squeeze(), joint_velocity.squeeze(),
                                                                contact.cpu(), acc.cpu().squeeze())
        return step_p, step_t