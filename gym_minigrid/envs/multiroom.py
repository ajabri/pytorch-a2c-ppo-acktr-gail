from gym_minigrid.minigrid import *
from gym_minigrid.register import register
from pdb import set_trace as st
import numpy as np

def manhattan_distance(pos1, pos2):
    # assume they are tuples
    return np.sum(np.abs(pos1[0] - pos2[0]) + np.abs(pos1[1] - pos2[1]))

class Room:
    def __init__(self,
        top,
        size,
        entryDoorPos,
        exitDoorPos,
        is_the_last_room = False,
    ):
        self.top = top
        self.size = size
        self.entryDoorPos = entryDoorPos
        self.exitDoorPos = exitDoorPos
        self.is_the_last_room = is_the_last_room

    def dis_to_sub_goal(self, pos, goal_pos):
        # In each room, the minimal distance to the exit is just the manhattan
        # between the current position and the exit
        if self.is_the_last_room:
            return manhattan_distance(goal_pos, pos)
        else:
            return manhattan_distance(self.exitDoorPos, pos)

    def entry_to_goal(self, goal_pos):
        if self.is_the_last_room:
            # assert goal_pos != None
            return manhattan_distance(self.entryDoorPos, goal_pos)
        else:
            return manhattan_distance(self.entryDoorPos, self.exitDoorPos)

    def in_room(self, pos):
        left, right, top, bottom = self.top[0], self.top[0] + self.size[0], self.top[1], self.top[1] + self.size[1]
        left_right = (pos[0] >= left and pos[0] <= right)
        up_down = (pos[1] >= top and pos[1] <= bottom)
        return (left_right and up_down)


class MultiRoomEnv(MiniGridEnv):
    """
    Environment with multiple rooms (subgoals)
    """

    def __init__(self,
        minNumRooms,
        maxNumRooms,
        maxRoomSize=10
    ):
        assert minNumRooms > 0
        assert maxNumRooms >= minNumRooms
        assert maxRoomSize >= 4

        self.minNumRooms = minNumRooms
        self.maxNumRooms = maxNumRooms
        self.maxRoomSize = maxRoomSize

        self.rooms = []

        super(MultiRoomEnv, self).__init__(
            grid_size=25,
            max_steps=self.maxNumRooms * 20,
            see_through_walls = True,
        )

        # TODO: change this
        self.reward_range = (-625, 0)

    def _gen_grid(self, width, height):
        roomList = []

        # Choose a random number of rooms to generate
        numRooms = self._rand_int(self.minNumRooms, self.maxNumRooms+1)

        while len(roomList) < numRooms:
            curRoomList = []

            entryDoorPos = (
                self._rand_int(0, width - 2),
                self._rand_int(0, width - 2)
            )

            # Recursively place the rooms
            self._placeRoom(
                numRooms,
                roomList=curRoomList,
                minSz=4,
                maxSz=self.maxRoomSize,
                entryDoorWall=2,
                entryDoorPos=entryDoorPos,
            )

            if len(curRoomList) > len(roomList):
                roomList = curRoomList

        # Store the list of rooms in this environment
        assert len(roomList) > 0
        self.rooms = roomList

        # Create the grid
        self.grid = Grid(width, height)
        wall = Wall()

        prevDoorColor = None

        # For each room
        for idx, room in enumerate(roomList):

            topX, topY = room.top
            sizeX, sizeY = room.size

            # Draw the top and bottom walls
            for i in range(0, sizeX):
                self.grid.set(topX + i, topY, wall)
                self.grid.set(topX + i, topY + sizeY - 1, wall)

            # Draw the left and right walls
            for j in range(0, sizeY):
                self.grid.set(topX, topY + j, wall)
                self.grid.set(topX + sizeX - 1, topY + j, wall)

            # If this isn't the first room, place the entry door
            if idx > 0:
                # Pick a door color different from the previous one
                doorColors = set(COLOR_NAMES)
                if prevDoorColor:
                    doorColors.remove(prevDoorColor)
                # Note: the use of sorting here guarantees determinism,
                # This is needed because Python's set is not deterministic
                doorColor = self._rand_elem(sorted(doorColors))

                entryDoor = Door(doorColor)
                self.grid.set(*room.entryDoorPos, entryDoor)
                prevDoorColor = doorColor

                prevRoom = roomList[idx-1]
                prevRoom.exitDoorPos = room.entryDoorPos

        # Randomize the starting agent position and direction
        self.place_agent(roomList[0].top, roomList[0].size)

        # Place the final goal in the last room
        self.goal_pos = self.place_obj(Goal(), roomList[-1].top, roomList[-1].size)

        self.mission = 'traverse the rooms to get to the goal'

    def _placeRoom(
        self,
        numLeft,
        roomList,
        minSz,
        maxSz,
        entryDoorWall,
        entryDoorPos,
    ):
        # Choose the room size randomly
        sizeX = self._rand_int(minSz, maxSz+1)
        sizeY = self._rand_int(minSz, maxSz+1)

        # The first room will be at the door position
        if len(roomList) == 0:
            topX, topY = entryDoorPos
        # Entry on the right
        elif entryDoorWall == 0:
            topX = entryDoorPos[0] - sizeX + 1
            y = entryDoorPos[1]
            topY = self._rand_int(y - sizeY + 2, y)
        # Entry wall on the south
        elif entryDoorWall == 1:
            x = entryDoorPos[0]
            topX = self._rand_int(x - sizeX + 2, x)
            topY = entryDoorPos[1] - sizeY + 1
        # Entry wall on the left
        elif entryDoorWall == 2:
            topX = entryDoorPos[0]
            y = entryDoorPos[1]
            topY = self._rand_int(y - sizeY + 2, y)
        # Entry wall on the top
        elif entryDoorWall == 3:
            x = entryDoorPos[0]
            topX = self._rand_int(x - sizeX + 2, x)
            topY = entryDoorPos[1]
        else:
            assert False, entryDoorWall

        # If the room is out of the grid, can't place a room here
        if topX < 0 or topY < 0:
            return False
        if topX + sizeX > self.width or topY + sizeY >= self.height:
            return False

        # If the room intersects with previous rooms, can't place it here
        for room in roomList[:-1]:
            nonOverlap = \
                topX + sizeX < room.top[0] or \
                room.top[0] + room.size[0] <= topX or \
                topY + sizeY < room.top[1] or \
                room.top[1] + room.size[1] <= topY

            if not nonOverlap:
                return False

        # Add this room to the list
        roomList.append(Room(
            (topX, topY),
            (sizeX, sizeY),
            entryDoorPos,
            None,
            is_the_last_room = (numLeft == 1),
        ))

        # If this was the last room, stop
        if numLeft == 1:
            return True

        # Try placing the next room
        for i in range(0, 8):

            # Pick which wall to place the out door on
            wallSet = set((0, 1, 2, 3))
            wallSet.remove(entryDoorWall)
            exitDoorWall = self._rand_elem(sorted(wallSet))
            nextEntryWall = (exitDoorWall + 2) % 4

            # Pick the exit door position
            # Exit on right wall
            if exitDoorWall == 0:
                exitDoorPos = (
                    topX + sizeX - 1,
                    topY + self._rand_int(1, sizeY - 1)
                )
            # Exit on south wall
            elif exitDoorWall == 1:
                exitDoorPos = (
                    topX + self._rand_int(1, sizeX - 1),
                    topY + sizeY - 1
                )
            # Exit on left wall
            elif exitDoorWall == 2:
                exitDoorPos = (
                    topX,
                    topY + self._rand_int(1, sizeY - 1)
                )
            # Exit on north wall
            elif exitDoorWall == 3:
                exitDoorPos = (
                    topX + self._rand_int(1, sizeX - 1),
                    topY
                )
            else:
                assert False

            # Recursively create the other rooms
            success = self._placeRoom(
                numLeft - 1,
                roomList=roomList,
                minSz=minSz,
                maxSz=maxSz,
                entryDoorWall=nextEntryWall,
                entryDoorPos=exitDoorPos
            )

            if success:
                break

        return True

    def step(self, action):
        self.step_count += 1

        # reward = 0
        done = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell == None or fwd_cell.can_overlap():
                self.agent_pos = fwd_pos
            if fwd_cell != None and fwd_cell.type == 'goal':
                done = True
                # reward = self._reward()
            if fwd_cell != None and fwd_cell.type == 'lava':
                done = True


        # Pick up an object
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(*fwd_pos, None)

        # Drop an object
        elif action == self.actions.drop:
            if not fwd_cell and self.carrying:
                self.grid.set(*fwd_pos, self.carrying)
                self.carrying.cur_pos = fwd_pos
                self.carrying = None

        # Toggle/activate an object
        elif action == self.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            assert False, "unknown action"

        reward = self._reward()
        if self.step_count >= self.max_steps:
            done = True

        obs = self.gen_obs()

        return obs, reward, done, {}

    def _reward(self):
        """
        Compute the reward to be given upon success
        """
        # msg = ''
        # for idx, room in enumerate(self.rooms):
        #     dist = room.entry_to_goal(goal_pos = self.goal_pos)
        #     msg += (" for room " + str(idx) + ", entry to exit: " + str(dist))
        dist = 0
        for idx, room in enumerate(self.rooms):
            # determine if the agent is in the room
            agent_in_room = room.in_room(self.agent_pos)
            if agent_in_room:
                # msg += (" agent in room " + str(idx+1) + " total number of rooms: " + str(len(self.rooms)))
                dist += room.dis_to_sub_goal(self.agent_pos, goal_pos = self.goal_pos)
                for remaining in range(idx+1, len(self.rooms)):
                    dist += room.entry_to_goal(goal_pos = self.goal_pos)
                break
        # msg += " distance: "+ str(dist) + " agent_pos: " + str(self.agent_pos) + " goal_pos: " + str(self.goal_pos)
        # print(msg)
        return -dist

class MultiRoomEnvN2S4(MultiRoomEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=2,
            maxNumRooms=2,
            maxRoomSize=4
        )


# class MultiRoomEnvN3S4(MultiRoomEnv):
#     def __init__(self):
#         super().__init__(
#             minNumRooms=3,
#             maxNumRooms=3,
#             maxRoomSize=4
#         )



class MultiRoomEnvN4S5(MultiRoomEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=4,
            maxNumRooms=4,
            maxRoomSize=5
        )

class MultiRoomEnvN6S6(MultiRoomEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=6,
            maxNumRooms=6,
            maxRoomSize=6
        )

class MultiRoomEnvN6(MultiRoomEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=6,
            maxNumRooms=6
        )

reward_threshold = 0.

register(
    id='MiniGrid-MultiRoom-N2-S4-v0',
    entry_point='gym_minigrid.envs:MultiRoomEnvN2S4',
    reward_threshold = reward_threshold,
)


# register(
#     id='MiniGrid-MultiRoom-N3-S4-v0',
#     entry_point='gym_minigrid.envs:MultiRoomEnvN3S4',
#     reward_threshold = reward_threshold,
# )

register(
    id='MiniGrid-MultiRoom-N4-S5-v0',
    entry_point='gym_minigrid.envs:MultiRoomEnvN4S5',
    reward_threshold = reward_threshold,
)

register(
    id='MiniGrid-MultiRoom-N6-v0',
    entry_point='gym_minigrid.envs:MultiRoomEnvN6',
    reward_threshold = reward_threshold,
)

register(
    id='MiniGrid-MultiRoom-N6-S6-v0',
    entry_point='gym_minigrid.envs:MultiRoomEnvN6S6',
    reward_threshold = reward_threshold,
)
