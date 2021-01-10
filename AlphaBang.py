import os
import pygame
import pygame.locals
import numpy as np
import copy
import time
import sys
import collections
import torch
import torch.nn.functional as F
import operator

"""
实现了AlphaZero，并且为价值网络引入Attention Mechanism以增强其全局关系建模能力
"""

WIDTH = 10
HEIGHT = 10
N_IN_ROW = 5   

class GoBang:
    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', WIDTH))
        self.height = int(kwargs.get('height', HEIGHT))
        self.n_in_row = int(kwargs.get('n_in_row', N_IN_ROW))
        self.board = None
        self.first_player = int(kwargs.get('first_player', 1))  # 1黑  -1白
        self.player = self.first_player
        self.wrong_move_reward = -1.
        self.observation = None

    def reset(self):
        self.player = self.first_player
        self.board = np.zeros((self.width, self.height))
        self.observation = self.board.copy()
        return self.transform([])  # 这里之所以是[]，因为其作为np.array索引的时候返回为空

    def step(self, action):
        """
        step to the next_observation
        首先判断落子是否合规
        若合规：
            判断落子位置有没有棋子：
                若有：
                    不改变状态，返回reward=self.wrong_move_reward,info=1表示表示落子地方有棋子
                若没有:
                    棋盘落子，判断是否棋局是否结束（平局或有胜负）从judge的返回接收done,reward

        若不合规：
            return KeyError

        :param action: tuple(int,int) or int
        :return: observation,reward,done,info
        """
        if type(action) is int:
            action = (action // self.width, action % self.width)
        assert type(action) == tuple
        info = None  # 暂时无用
        if 0 <= action[0] <= self.width - 1 and 0 <= action[1] <= self.height - 1:
            if abs(self.board[action]) > 0:
                raise Exception("location error")  # 动作落子位置有棋子
            else:
                self.board[action] = self.player
                self.player = -1 * self.player
                reward, done = self.determine_vectory(action)
                observation = self.transform(action)
                return observation, reward, done, info
        else:
            raise Exception("location error")  # action未在棋盘内

    def get_available_moves(self):
        """获取当前棋面的可落子的位置"""
        x, y = np.where(self.board == 0)
        return [int(i * self.width + j) for i, j in zip(x, y)]

    def transform(self, action):
        """
        将输入的observation转化为供policy_net学习的state
        输入action是step转化后的action，为tuple
        输出observation：np.array , shape=(4, WIDTH, HEIGHT)
        """
        observation = np.zeros((3, WIDTH, HEIGHT))  # 4层二值特征平面。第一层表示当前棋手棋子，
        # 第二层表示对手棋子，第三层表示上一手对手落子位置，第四层表示初始棋手
        observation[0][self.board == self.player] = 1.0
        observation[1][self.board == -self.player] = 1.0
        # observation[2][action] = 1.0
        observation[2] = (self.first_player == self.player)  # transform将为一个棋手
        self.observation = observation.copy()  # 和其执行一个动作后的结果结合
        return observation

    def determine_vectory(self, action):
        """
        判断棋盘是否满了
        若没满
            判断落子位置上下左右，左上，左下，右上，右下是否有self.n_in_row个连成一起棋子

        :param action: 落子位置
        :return: reward,done
        """
        reward = 0
        done = False
        if (self.board == 0).sum() == 0:
            reward = 0
            done = True
        else:
            possible_pieces_list = []
            up = max(action[0] - self.n_in_row + 1, 0)
            down = min(action[0] + self.n_in_row - 1, self.width - 1)
            left = max(action[1] - self.n_in_row + 1, 0)
            right = min(action[1] + self.n_in_row - 1, self.height - 1)
            left_up = min(min(action[0], action[1]), self.n_in_row - 1)
            right_down = min(min(self.width - action[0] - 1, self.height - action[1] - 1), self.n_in_row - 1)
            right_up = min(min(self.height - action[1] - 1, action[0]), self.n_in_row - 1)
            left_down = min(min(self.width - action[0] - 1, action[1]), self.n_in_row - 1)
            possible_pieces_list.append(list([self.board[i][action[1]] for i in \
                                              range(up, down + 1)]))
            possible_pieces_list.append(list([self.board[action[0]][i] for i in \
                                              range(left, right + 1)]))
            possible_pieces_list.append(list([self.board[action[0] - left_up + i] \
                                                  [action[1] - left_up + i] \
                                              for i in range(left_up + right_down + 1)]))
            possible_pieces_list.append(list([self.board[action[0] + left_down - \
                                                         i][action[1] - left_down + i] \
                                              for i in range(left_down + right_up + 1)]))
            for i in possible_pieces_list:
                for j in range(len(i) - self.n_in_row + 1):
                    if abs(sum(i[(0 + j):(self.n_in_row + j)])) == self.n_in_row:
                        reward = 1
                        done = True
                        break

        return reward, done

    def render(self):
        pass

    def __str__(self):
        return str(self.board.copy())


class Player:
    def __init__(self,
                 who_play,
                 init_player,
                 env=None,
                 mcts=None,
                 pure_mcts=None,
                 c_puct=5.,
                 noise=1):
        self.who_play = who_play  # 0:human   1:agent  2:uct
        self.history = []
        self.env = env
        self.mcts = mcts
        self.pure_mcts = pure_mcts
        self.init_player = init_player
        self.c_puct = c_puct
        self.noise = noise

    def get_action(self, ):
        if self.who_play == 0:
            a = True
            while a:
                for event in pygame.event.get():
                    if event.type == pygame.locals.QUIT:
                        pygame.quit()
                        sys.exit()
                    else:
                        if event.type == pygame.locals.MOUSEBUTTONUP and event.button == 1:
                            position = self.get_position(event.pos)
                            print(event.pos, position)
                            action = position
                            a = False
                            break

        else:
            action = self.machine_player()
        return action

    def machine_player(self):
        if self.who_play == 1:
            action, mcts_prob = self.mcts.run_mcts(self.env, self.c_puct, self.noise)
        elif self.who_play == 2:
            action = self.pure_mcts.run_mcts(self.env, c_puct=self.c_puct)
        return action

    @staticmethod
    def get_position(position):
        x = (position[0] - 1 * GoBangBoard.UNIT_PIXEL) // GoBangBoard.UNIT_PIXEL
        y = (position[1] - 2 * GoBangBoard.UNIT_PIXEL) // GoBangBoard.UNIT_PIXEL
        return y, x


class GoBangBoard:
    UNIT_PIXEL = 30
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 0)

    def __init__(self, env):
        self.gobang = env
        self.board = None
        pygame.init()

    def draw(self, done=False):
        color_dict = {1: GoBangBoard.BLACK, -1: GoBangBoard.WHITE}
        player_do_dict = {1: "请黑方行棋", -1: "请白方行棋"}
        player_do = player_do_dict[self.gobang.player]
        self.board = pygame.display.set_mode(((self.gobang.height + 2) * GoBangBoard.UNIT_PIXEL,
                                              (self.gobang.width + 3) * GoBangBoard.UNIT_PIXEL))
        pygame.display.set_caption('GoBang')
        self.board.fill(GoBangBoard.YELLOW)
        for i in range(self.gobang.width + 1):
            pygame.draw.line(self.board,
                             GoBangBoard.BLACK,
                             (1 * GoBangBoard.UNIT_PIXEL, (i + 2) * GoBangBoard.UNIT_PIXEL),
                             ((1 + self.gobang.height) * GoBangBoard.UNIT_PIXEL, (i + 2) * GoBangBoard.UNIT_PIXEL),
                             )
        for i in range(self.gobang.height + 1):
            pygame.draw.line(self.board,
                             GoBangBoard.BLACK,
                             ((1 + i) * GoBangBoard.UNIT_PIXEL, 2 * GoBangBoard.UNIT_PIXEL),
                             ((1 + i) * GoBangBoard.UNIT_PIXEL, (2 + self.gobang.width) * GoBangBoard.UNIT_PIXEL))
        for i in range(self.gobang.width * self.gobang.height):
            x = i % self.gobang.width
            y = i // self.gobang.width
            if self.gobang.board[x][y]:
                color = color_dict[self.gobang.board[x][y]]
                pos = (int((y + 1.5) * GoBangBoard.UNIT_PIXEL), int((x + 2.5) * GoBangBoard.UNIT_PIXEL))
                pygame.draw.circle(self.board,
                                   color,
                                   pos,
                                   int(0.4 * GoBangBoard.UNIT_PIXEL))
        if done:
            fontObj = pygame.font.SysFont('SimHei', 24)
            textSurfaceObj = fontObj.render('Game Over', True, GoBangBoard.BLACK, GoBangBoard.YELLOW)
            RecObj = textSurfaceObj.get_rect()
            RecObj.center = (5 * GoBangBoard.UNIT_PIXEL, 1 * GoBangBoard.UNIT_PIXEL)
            self.board.blit(textSurfaceObj, RecObj)
        else:
            fontObj = pygame.font.SysFont('SimHei', 24)
            textSurfaceObj = fontObj.render(player_do, True, GoBangBoard.BLACK, GoBangBoard.YELLOW)
            RecObj = textSurfaceObj.get_rect()
            RecObj.center = (5 * GoBangBoard.UNIT_PIXEL, 1 * GoBangBoard.UNIT_PIXEL)
            self.board.blit(textSurfaceObj, RecObj)
        pygame.display.update()
        return

    def run(self, player1, player2, ):
        """
        player1,player2:player class
        n_player:0-2,int
        :param player1:
        :param kwargs:
        :return:
        """
        first_player, second_player = (player1, player2) if player1.init_player == 1 else \
            (player2, player1)
        done = False
        winners = []
        self.gobang.reset()
        while True:
            self.draw(done)
            action = first_player.get_action()
            observation, reward, done, info = self.gobang.step(action)
            self.draw(done)
            if done:
                winners.append(-1 * self.gobang.player)
                again = input("Do you want to play again?(yes/no)")
                if again.lower() == 'yes':
                    self.gobang.reset()
                    done = False
                    continue
                else:
                    break
            action = second_player.get_action()
            observation, reward, done, info = self.gobang.step(action)
            self.draw(done)
            if done:
                winners.append(-1 * self.gobang.player)
                again = input("Do you want to play again?(yes/no)")
                if again.lower() == 'yes':
                    self.gobang.reset()
                    done = False
                    continue
                else:
                    break
        return winners


class Replay_Buffer:
    def __init__(self, buffer_capacity=100000):
        self.buffer = collections.deque(maxlen=buffer_capacity)
        self.buffer_capacity = buffer_capacity

    def store(self, observation, mcts_prob, reward):
        """
        存储self-play结果
        @param observation: np.array , shape=(4, WIDTH, HEIGHT)
        @param mcts_prob: np.array , shape=(WIDTH*HEIGHT,)
        @param reward:    float,scalar
        """
        self.buffer.append([observation, mcts_prob, reward])

    def random_sample(self, batch_size):
        """
        随机采样样本
        @param batch_size: 样本数
        @return: observations,mcts_probs,rewards都分别是observation,mcts_prob,reward的集合
                之所以如此是方便torch.tensor转换的时候转换为batch_size×相应的shape
        """
        indices = np.random.choice(range(len(self.buffer)), batch_size)
        observations = [self.buffer[i][0] for i in indices]
        mcts_probs = [self.buffer[i][1] for i in indices]
        rewards = [self.buffer[i][2] for i in indices]
        return observations, mcts_probs, rewards

    def buffer_size(self):
        return len(self.buffer)


class Policy_Net(torch.nn.Module):
    def __init__(self):
        super(Policy_Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.act_conv1 = torch.nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = torch.nn.Linear(4 * WIDTH * HEIGHT, WIDTH * HEIGHT)

        self.val_conv1 = torch.nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = torch.nn.Linear(2 * WIDTH * HEIGHT, 64)
        self.val_fc2 = torch.nn.Linear(64, 1)

    def forward(self, state_input):
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4 * WIDTH * HEIGHT)
        x_log_act = F.log_softmax(self.act_fc1(x_act), dim=-1)

        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2 * WIDTH * HEIGHT)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = torch.tanh(self.val_fc2(x_val))
        return x_val, x_log_act


class NL(torch.nn.Module):
    """
    y = normalize(f(x))g(x) f(x)是计算相似度，g(x)为线性变换.此处引用nlp中self-attention的q,k,v,
    q,k为计算f(x),v计算g(x)

    z = w·y+x
    """

    def __init__(self, in_channels, out_channels=None, dimension=2,
                 sub_sample=False, bn_layer=True):

        super(NL, self).__init__()
        assert dimension in [1, 2, 3]
        self.in_channels = in_channels
        if out_channels is None:
            self.out_channels = in_channels // 2
            # self.out_channels = in_channels
        else:
            self.out_channels = out_channels
        self.dimension = dimension

        if dimension == 1:
            Conv = torch.nn.Conv1d
            maxpooling = torch.nn.MaxPool1d(kernel_size=(2))
            bn = torch.nn.BatchNorm1d
        elif dimension == 2:
            Conv = torch.nn.Conv2d
            maxpooling = torch.nn.MaxPool2d(kernel_size=(2, 2))
            bn = torch.nn.BatchNorm2d
        else:
            Conv = torch.nn.Conv3d
            maxpooling = torch.nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = torch.nn.BatchNorm3d

        self.q = Conv(in_channels=self.in_channels, out_channels=self.out_channels,
                      kernel_size=1, stride=1, padding=0)
        self.k = Conv(in_channels=self.in_channels, out_channels=self.out_channels,
                      kernel_size=1, stride=1, padding=0)
        self.v = Conv(in_channels=self.in_channels, out_channels=self.out_channels,
                      kernel_size=1, stride=1, padding=0)
        if sub_sample:
            self.k = torch.nn.Sequential(
                self.k,
                maxpooling
            )
            self.v = torch.nn.Sequential(
                self.v,
                maxpooling
            )

        self.w = Conv(in_channels=self.out_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0)
        if bn_layer:
            self.w = torch.nn.Sequential(self.w,
                                         bn(self.in_channels))

    def forward(self, x):
        batch_size = x.shape[0]

        k = self.k(x).view(batch_size, self.out_channels, -1)
        k = k.permute(0, 2, 1).contiguous()

        q = self.q(x).view(batch_size, self.out_channels, -1)

        v = self.v(x).view(batch_size, self.out_channels, -1)

        f = torch.matmul(k,q)
        f_normalized = torch.softmax(f, dim=-1)

        y = torch.matmul(v,f_normalized)
        y = y.view(batch_size, self.out_channels, *x.size()[2:])
        z = self.w(y) + x
        return z


class NonLocal2D(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 reduction=2,
                 use_scale=True,
                 mode='embedded_gaussian',
                 whiten_type='channel',
                 downsample=False,
                 ):
        super(NonLocal2D, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.use_scale = use_scale
        self.inter_channels = in_channels // reduction
        self.mode = mode
        assert mode in ['embedded_gaussian', 'dot_product', 'gaussian']
        if mode == 'gaussian':
            self.with_embedded = False
        else:
            self.with_embedded = True
        self.whiten_type = whiten_type
        assert whiten_type in [None, 'channel', 'bn-like']

        if downsample:
            self.downsample = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        else:
            self.downsample = None

        self.g = torch.nn.Conv2d(
            self.in_channels,
            self.inter_channels,
            kernel_size=1,
        )
        if self.with_embedded:
            self.theta = torch.nn.Conv2d(
                self.in_channels,
                self.inter_channels,
                kernel_size=1,
            )
            self.phi = torch.nn.Conv2d(
                self.in_channels,
                self.inter_channels,
                kernel_size=1,
            )
        self.conv_out = torch.nn.Conv2d(
            self.inter_channels,
            self.in_channels,
            kernel_size=1,
        )

    def forward(self, x):
        n, _, h, w = x.shape
        if self.downsample:
            down_x = self.downsample(x)
        else:
            down_x = x

        g_x = self.g(down_x).view(n, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        if self.with_embedded:
            theta_x = self.theta(x).view(n, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
        else:
            theta_x = x.view(n, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)

        if self.with_embedded:
            phi_x = self.phi(down_x).view(n, self.inter_channels, -1)
        else:
            phi_x = x.view(n, self.in_channels, -1)

        # whiten
        if self.whiten_type == "channel":
            theta_x_mean = theta_x.mean(2).unsqueeze(2)
            phi_x_mean = phi_x.mean(2).unsqueeze(2)
            theta_x -= theta_x_mean
            phi_x -= phi_x_mean
        elif self.whiten_type == 'bn-like':
            theta_x_mean = theta_x.mean(2).mean(0).unsqueeze(0).unsqueeze(2)
            phi_x_mean = phi_x.mean(2).mean(0).unsqueeze(0).unsqueeze(2)
            theta_x -= theta_x_mean
            phi_x -= phi_x_mean

        pairwise_weight = torch.matmul(theta_x, phi_x)

        y = torch.matmul(pairwise_weight, g_x)
        y = y.permute(0, 2, 1).reshape(n, self.inter_channels, h, w)

        output = x + self.conv_out(y)

        return output

class DNL(torch.nn.Module):
    """
    y = normalize(f(x))g(x) f(x)是计算相似度，g(x)为线性变换.此处引用nlp中self-attention的q,k,v,
    q,k为计算f(x),v计算g(x)

    后续加了disentangled self-attention model 所以有k_pairwise , k_unary
    为了去耦合，k_unary没有和q的u相乘了，然而expend成pairwise项想乘的size进行运算。
    z = w·y+x
    """
    def __init__(self,in_channels,
                 out_channels=None,
                 dimension=2,
                 bn_layer=True):

        super(DNL,self).__init__()
        assert dimension in [1,2,3]
        self.in_channels = in_channels
        if out_channels is None:
            self.out_channels = in_channels
        else:
            self.out_channels = out_channels
        self.dimension = dimension

        if dimension==1:
            Conv = torch.nn.Conv1d
            maxpooling = torch.nn.MaxPool1d(kernel_size=(2))
            bn = torch.nn.BatchNorm1d
        elif dimension==2:
            Conv = torch.nn.Conv2d
            maxpooling = torch.nn.MaxPool2d(kernel_size=(2,2))
            bn = torch.nn.BatchNorm2d
        else:
            Conv = torch.nn.Conv3d
            maxpooling = torch.nn.MaxPool3d(kernel_size=(1,2,2))
            bn = torch.nn.BatchNorm3d

        self.q = Conv(in_channels=self.in_channels,out_channels=self.out_channels,
                      kernel_size=1,stride=1,padding=0)
        self.k = Conv(in_channels=self.in_channels,out_channels=self.out_channels,
                      kernel_size=1,stride=1,padding=0)
        self.m = Conv(in_channels=self.in_channels, out_channels=1,
                               kernel_size=1, stride=1, padding=0)
        self.v = Conv(in_channels=self.in_channels,out_channels=self.out_channels,
                      kernel_size=1,stride=1,padding=0)
        self.w = Conv(in_channels=self.in_channels,out_channels=self.out_channels,
                  kernel_size=1,stride=1,padding=0)
        if bn_layer:
            self.w = torch.nn.Sequential(self.w,
                                         bn(self.out_channels))


    def forward(self,x):
        batch_size = x.shape[0]

        k = self.k(x)
        with torch.no_grad():
            k_u = torch.mean(k,dim=(-2,-1),keepdim=True).expand_as(k)
        k_whiten = (k-k_u).view(batch_size,self.out_channels,-1)
        k_whiten = k_whiten.permute(0,2,1).contiguous()

        q = self.q(x)
        with torch.no_grad():
            q_u = torch.mean(q, dim=(-2, -1), keepdim=True).expand_as(q)
        q_whiten = (q - q_u)
        q_whiten = q_whiten.view(batch_size, self.out_channels, -1)

        qk = torch.matmul(k_whiten, q_whiten)

        m = self.m(x).view(batch_size,1,-1)
        m = m.permute(0,2,1).contiguous()
        m = torch.matmul(m,m.permute(0,2,1).contiguous())

        v = self.v(x).view(batch_size,self.out_channels,-1)

        f = torch.softmax(qk,dim=-1) + torch.softmax(m,dim=0)
        y = torch.matmul(v,f)
        y = y.view(batch_size, self.out_channels, *x.size()[2:])
        z = y + self.w(x)
        return z



class Policy_Net_With_Attention(torch.nn.Module):
    def __init__(self):
        super(Policy_Net_With_Attention, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.attention1 = NonLocal2D(32)
        # self.attention2 = NL(64)
        # self.attention = DNL(3)
        self.act_conv1 = torch.nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = torch.nn.Linear(4 * WIDTH * HEIGHT, WIDTH * HEIGHT)

        self.val_conv1 = torch.nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = torch.nn.Linear(2 * WIDTH * HEIGHT, 64)
        self.val_fc2 = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.attention1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4 * WIDTH * HEIGHT)
        x_act = F.log_softmax(self.act_fc1(x_act), dim=-1)

        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2 * WIDTH * HEIGHT)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = torch.tanh(self.val_fc2(x_val))
        return x_val, x_act


class Policy_Net_With_DNL(torch.nn.Module):
    def __init__(self):
        super(Policy_Net_With_DNL, self).__init__()
        self.dnl1 = DNL(32,32)
        self.dnl2 = DNL(64,64)
        # self.dnl3 = DNL(64,128)
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.act_conv1 = torch.nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = torch.nn.Linear(4 * WIDTH * HEIGHT, WIDTH * HEIGHT)

        self.val_conv1 = torch.nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = torch.nn.Linear(2 * WIDTH * HEIGHT, 64)
        self.val_fc2 = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dnl1(x)
        x = F.relu(self.conv2(x))
        x = self.dnl2(x)
        x = F.relu(self.conv3(x))

        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4 * WIDTH * HEIGHT)
        x_act = F.log_softmax(self.act_fc1(x_act), dim=-1)

        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2 * WIDTH * HEIGHT)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = torch.tanh(self.val_fc2(x_val))
        return x_val, x_act

class TreeNode:
    def __init__(self, parent, prior_prob):
        self.parent = parent
        self.children = {}
        self.prior_prob = prior_prob
        self.n_visits = 0
        self.q = 0

    def select(self, c_puct=5.):
        """
        进行节点选择
        @param c_puct: 控制
        @return:
        """
        move_node = None

        if self.children.items():
            moves_nodes = list(self.children.items())
            # with torch.no_grad():
            #     q = torch.tensor([node[1].q for node in moves_nodes])
            #     prior_prob = torch.tensor([node[1].prior_prob for node in moves_nodes])
            #     parent_n_visit = torch.tensor([node[1].parent.n_visits for node in moves_nodes])
            #     n_visit = n_visit = torch.tensor([node[1].n_visits for node in moves_nodes])
            #     prob_noise = torch.distributions.dirichlet.Dirichlet(prior_prob).sample()
            #     q_u = q + c_puct * (0.75 * prior_prob + 0.25 * prob_noise) * \
            #             torch.sqrt(parent_n_visit)/(n_visit + 1)
            # 在expand处的prob加上噪声更方便
            q_u = [node[1].q + c_puct * node[1].prior_prob * \
                   np.sqrt(node[1].parent.n_visits) / (node[1].n_visits + 1)
                   for node in moves_nodes]
            # print(q_u)
            index = int(np.argmax(q_u))
            move_node = moves_nodes[index]
        return move_node

    def update(self, v):
        """
        更新节点的访问次数和平均价值q
        @param v: value
        @return:
        """
        self.n_visits += 1
        self.q += 1.0 * (v - self.q) / self.n_visits

    def update_recursive(self, v):
        """
        递归地更新该路径上的节点的信息，注意由于是self-play，所以递归调用父节点的更新时候要取反
        @param v:
        @return:
        """
        if self.parent is not None:
            self.parent.update_recursive(-v)
        self.update(v)


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class MCTS:
    def __init__(self, policy_net, n_playout=400, ):
        self.root = None
        self.policy_net = policy_net
        self.n_playout = n_playout

    def run_mcts(self, env, c_puct=5., noise=1, new_game=True):
        """
        执行mcts选择动作
        @param env: 当前env，后面用的其复制体
        @param c_puct: 控制node.select时探索利用平衡的参数
        @param noise:  控制mcts实际选择动作时候的方式
        @return: action 为 int scalar ,mcts_probs为np.array shape=(WIDTH*HEIGHT,)
        """
        if new_game:
            self.root = TreeNode(None, None)
        for i in range(self.n_playout):
            env_copy = copy.deepcopy(env)
            observation = env_copy.observation
            node = self.root
            while True:
                select_return = node.select(c_puct=c_puct)
                if select_return is None:
                    available_moves = env_copy.get_available_moves()
                    self.expand(node, observation, available_moves)
                    break
                else:
                    move, node = select_return
                    observation, reward, done, info = env_copy.step(move)
                    if done:
                        if reward == 1:
                            node.update_recursive(1.)  # 如果叶节点是对局结束局面，
                            # 则回传由胜利带来的v,自定义，论文似乎不回传
                        break
        move, nodes = zip(*self.root.children.items())
        n_visits = [i.n_visits for i in nodes]
        if noise == 1:
            action_probs = softmax(np.log(np.array(n_visits)+1e-6) + 1e-6)
            action_probs = 0.8 * action_probs + \
                           0.2 * np.random.dirichlet(0.3 * np.ones_like(action_probs))
            action_index = np.random.choice(range(len(action_probs)),
                                            p=action_probs)
            action = move[action_index]
        else:
            action_probs = np.zeros((len(n_visits)))
            action_probs[np.argmax(n_visits)] = 1.
            action_index = np.argmax(n_visits)
            action = move[action_index]
        mcts_probs = np.zeros(WIDTH * HEIGHT)  # 注意此处不能是(WIDTH*HEIGHT,1)
        mcts_probs[list(move)] = action_probs
        self.update_root(nodes[action_index])
        return action, mcts_probs

    def update_root(self, action_node):
        self.root = action_node
        self.root.parent = None

    def expand(self, node, observation, available_moves):
        """
        对叶子节点进行拓展
        @param node: 叶子节点
        @param observation:叶子节点对应的局面
        @param available_moves:叶子节点对应的对手的可行子位置
        @return:
        """
        observation = torch.tensor(observation[np.newaxis], dtype=torch.float32)
        value, prior_log_prob = self.policy_net(observation)
        prior_prob = torch.exp(prior_log_prob)
        value = value.item()
        prior_prob = prior_prob.data.numpy()

        for i in available_moves:
            if i not in node.children.keys():
                node.children[i] = TreeNode(node, prior_prob[0][i])
            else:
                continue
        if node.parent is not None:
            node.update_recursive(-value)
        return


def policy_fn(available_moves):
    a = torch.zeros((1, WIDTH * HEIGHT))
    a[0][available_moves] = 1. / len(available_moves)
    return 0., a


def mcts_leaf_evaluate(env, move, player):
    while True:
        observation, reward, done, info = env.step(move)
        if done:
            reward = 1 if -1 * env.player == player else -1
            break
        available_moves = env.get_available_moves()
        move = np.random.choice(available_moves, 1).item()
    return reward


class Pure_MCTS():
    def __init__(self, policy_fn, n_playout=1000, ):
        self.root = None
        self.policy_net = policy_fn
        self.n_playout = n_playout

    def run_mcts(self, env, c_puct=5., temp=0, ):
        """
        执行mcts选择动作
        @param env: 当前env，后面用的其复制体
        @param c_puct: 控制node.select时探索利用平衡的参数
        @param temp:  控制mcts实际选择动作时候的方式
        @return: action 为 int scalar ,mcts_probs为np.array shape=(WIDTH*HEIGHT,)
        """

        self.root = TreeNode(None, None)
        for i in range(self.n_playout):
            env_copy = copy.deepcopy(env)
            node = self.root
            while True:
                select_return = node.select(c_puct=c_puct)
                if select_return is None:
                    available_move = env_copy.get_available_moves()
                    self.expand(node, available_move)
                    move = available_move[np.argmax(list(node.children.keys())).item()]
                    current_player = env_copy.player
                    leaf_value = mcts_leaf_evaluate(env_copy, move, current_player)
                    node.children[move].update_recursive(leaf_value)
                    break
                else:
                    move, node = select_return
                    observation, reward, done, info = env_copy.step(move)
                    if done:
                        if reward == 1:
                            node.update_recursive(1.)
                        else:
                            pass
                            # node.update_recursive(-1.)
                        break
        move, node = zip(*self.root.children.items())
        n_visits = [i.n_visits for i in node]
        if temp == 1:
            action_probs = F.softmax(torch.tensor(n_visits, dtype=torch.float32), dim=-1)
            action_probs = 0.98 * action_probs + \
                           0.02 * torch.distributions.dirichlet.Dirichlet(action_probs).sample()
            action_index = torch.multinomial(action_probs, 1).item()
            action = move[action_index]
        else:
            action_probs = np.zeros((len(n_visits)))
            action_probs[np.argmax(n_visits)] = 1.
            action_index = np.argmax(n_visits)
            action = move[action_index]

        return action

    def expand(self, node, availabel_moves):
        """
        对叶子节点进行拓展
        @param node: 叶子节点
        @param observation:叶子节点对应的局面
        @param availabel_move:叶子节点对应的对手的可行子位置
        @return:
        """
        _, prior_prob = self.policy_net(availabel_moves)

        prior_prob = prior_prob.data.numpy()

        for i in availabel_moves:
            if i not in node.children.keys():
                node.children[i] = TreeNode(node, prior_prob[0][i])
            else:
                continue
        return


class AlphaGoBang:
    def __init__(self,
                 env,
                 mcts,
                 replay_buffer,
                 policy_net,
                 optimizer,
                 lr=3e-4,
                 batch_size=512,
                 batches=10,
                 least_samples_to_sample=5000
                 ):
        self.env = env
        self.policy_net = policy_net
        self.mcts = mcts
        self.replay_buffer = replay_buffer
        self.optimizer = optimizer
        self.lr = lr
        self.batch_size = batch_size
        self.batches = batches
        self.least_samples_to_sample = least_samples_to_sample
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

    def self_play(self, c_puct=5., noise=1):
        observation = self.env.reset()
        store_history = []
        new_game = True
        while True:
            action, mcts_probs = self.mcts.run_mcts(self.env, c_puct, noise, new_game)
            new_game = False
            store_history.append([observation, mcts_probs])
            observation, reward, done, info = self.env.step(action)
            if done:
                break
        if reward == 1:
            winner = -1 * self.env.player
            reward = 1.
        else:
            winner = 0  # 表示平局
            reward = 0.
        play_data = []
        for i in store_history[::-1]:
            play_data.append([i[0], i[1], reward])
            reward = -1. * reward
        self.expand_data_and_store(play_data)
        value_loss, policy_loss = None, None
        if len(self.replay_buffer.buffer) > self.least_samples_to_sample:
            for i in range(self.batches):
                value_loss, policy_loss = self.train()
        return winner, value_loss, policy_loss

    def expand_data_and_store(self, play_data):
        """augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        for data in play_data:
            observation, mcts_probs, reward = data
            mcts_probs = mcts_probs.reshape((WIDTH, HEIGHT))
            for i in [1, 2, 3, 4]:
                expand_observation = np.array([np.rot90(obs, i) for obs in observation])
                expand_mcts_probs = np.rot90(mcts_probs, i)
                self.replay_buffer.store(expand_observation,
                                         expand_mcts_probs.flatten(),
                                         reward)
            expand_observation = np.array([np.fliplr(obs) for obs in observation])
            expand_mcts_probs = np.fliplr(mcts_probs)
            self.replay_buffer.store(expand_observation,
                                     expand_mcts_probs.flatten(),
                                     reward)

    def train(self):
        observations, mcts_probs, rewards = self.replay_buffer.random_sample(self.batch_size)
        observations = torch.tensor(observations, dtype=torch.float32)
        mcts_probs = torch.tensor(mcts_probs, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)

        self.optimizer.zero_grad()
        pre_values, pre_log_probs = self.policy_net(observations)
        value_loss = F.mse_loss(rewards, pre_values.view(-1))
        policy_loss = -1. * torch.mean((pre_log_probs * mcts_probs).sum(dim=-1))
        loss = value_loss + policy_loss
        loss.backward()
        self.optimizer.step()

        return value_loss.data, policy_loss.data


def test(pure_mcts, mcts, test_env, n=20):
    print("begin to test:", end="")
    test_winners = []
    for i in range(n):
        test_env.reset()
        while True:
            action, mcts_probs = mcts.run_mcts(test_env, c_puct=5., noise=0)
            observation, reward, done, info = test_env.step(action)
            if done:
                test_winners.append(1)
                break
            action = pure_mcts.run_mcts(test_env, c_puct=5.)
            observation, reward, done, info = test_env.step(action)
            if done:
                test_winners.append(0)
                break
    return np.sum(test_winners) / n


train_alpha = 0
load_flag = 0
model_file = "AlphaBang.pkl"
if train_alpha:
    n_playout = 1000
    env = GoBang(first_player=1)
    test_env = GoBang(first_player=1)
    policy_net = Policy_Net_With_Attention()
    # policy_net = Policy_Net_With_DNL()
    # policy_net = Policy_Net()
    mcts = MCTS(policy_net=policy_net,
                n_playout=400, )
    pure_mcts = Pure_MCTS(policy_fn=policy_fn,
                          n_playout=n_playout)
    replay_buffer = Replay_Buffer(buffer_capacity=100000)
    optimizer = torch.optim.Adam(policy_net.parameters(),
                                 weight_decay=1e-4)
    if os.path.exists(model_file) and load_flag == 1:
        alphagobang = torch.load(model_file)
        print("load alphabang model successful")
    else:
        alphagobang = AlphaGoBang(env=env,
                                  mcts=mcts,
                                  replay_buffer=replay_buffer,
                                  policy_net=policy_net,
                                  optimizer=optimizer,
                                  least_samples_to_sample=3000,
                                  batches=5,
                                  batch_size=512,
                                  lr=3e-4)
        print('init alphabang successful')
    c_puct = 5.
    noise = 1
    winners, value_losses, policy_losses = [], [], []
    for i in range(1000):
        print(i)
        winner, value_loss, policy_loss = alphagobang.self_play(c_puct=c_puct,
                                                                noise=noise)
        winners.append(winner)
        value_losses.append(value_loss)
        policy_losses.append(policy_loss)
        if i % 100 == 0 and i!=0:
            win_rate = test(pure_mcts, mcts, test_env, 10)
            print(win_rate)
    torch.save(alphagobang, model_file)
else:
    n_playout1 = 400
    n_playout2 = 1000
    who_player1 = 1
    who_player2 = 0
    init_player = 1
    init_player1, init_player2 = (1, 2) if init_player == 1 else (2, 1)
    c_puct1 = 5.
    c_puct2 = 5.
    noise = 0
    env = GoBang()
    try:
        policy_net = torch.load(model_file).policy_net
    except:
        policy_net = None
        print("No such file or directory: 'AlphaGoBang.pkl'")
        pass
    mcts1 = MCTS(policy_net=copy.deepcopy(policy_net),
                 n_playout=n_playout1)
    pure_mcts1 = Pure_MCTS(policy_fn=policy_fn,
                           n_playout=n_playout1)
    mcts2 = MCTS(policy_net=policy_net,
                 n_playout=n_playout2)
    pure_mcts2 = Pure_MCTS(policy_fn=policy_fn,
                           n_playout=n_playout2)
    player1 = Player(who_play=who_player1,
                     init_player=init_player1,
                     env=env,
                     mcts=mcts1,
                     pure_mcts=pure_mcts1,
                     c_puct=c_puct1,
                     noise=noise)
    player2 = Player(who_play=who_player2,
                     init_player=init_player1,
                     env=env,
                     mcts=mcts2,
                     pure_mcts=pure_mcts2,
                     c_puct=c_puct2,
                     noise=noise)
    gobangboard = GoBangBoard(env)
    winers = gobangboard.run(player1, player2)
