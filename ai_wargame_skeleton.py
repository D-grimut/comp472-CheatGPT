from __future__ import annotations
import argparse
import copy
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from time import sleep
from typing import Tuple, TypeVar, Type, Iterable, ClassVar
import random
import requests

# maximum and minimum values for our heuristic scores (usually represents an end of game condition)
MAX_HEURISTIC_SCORE = 2000000000
MIN_HEURISTIC_SCORE = -2000000000


class UnitType(Enum):
    """Every unit type."""

    AI = 0
    Tech = 1
    Virus = 2
    Program = 3
    Firewall = 4


class MoveType(Enum):
    """Every Move Type - for move validation."""

    Invalid = -1
    Repair = 1
    Attack = 2
    Advance = 3
    SelfDestruct = 4


class Player(Enum):
    """The 2 players."""

    Attacker = 0
    Defender = 1

    def next(self) -> Player:
        """The next (other) player."""
        if self is Player.Attacker:
            return Player.Defender
        else:
            return Player.Attacker


class GameType(Enum):
    AttackerVsDefender = 0
    AttackerVsComp = 1
    CompVsDefender = 2
    CompVsComp = 3


##############################################################################################################


@dataclass(slots=True)
class Unit:
    player: Player = Player.Attacker
    type: UnitType = UnitType.Program
    health: int = 9
    # class variable: damage table for units (based on the unit type constants in order)
    damage_table: ClassVar[list[list[int]]] = [
        [3, 3, 3, 3, 1],  # AI
        [1, 1, 6, 1, 1],  # Tech
        [9, 6, 1, 6, 1],  # Virus
        [3, 3, 3, 3, 1],  # Program
        [1, 1, 1, 1, 1],  # Firewall
    ]

    # class variable: repair table for units (based on the unit type constants in order)
    repair_table: ClassVar[list[list[int]]] = [
        [0, 1, 1, 0, 0],  # AI
        [3, 0, 0, 3, 3],  # Tech
        [0, 0, 0, 0, 0],  # Virus
        [0, 0, 0, 0, 0],  # Program
        [0, 0, 0, 0, 0],  # Firewall
    ]

    def is_alive(self) -> bool:
        """Are we alive ?"""
        return self.health > 0

    def mod_health(self, health_delta: int):
        """Modify this unit's health by delta amount."""
        self.health += health_delta
        if self.health < 0:
            self.health = 0
        elif self.health > 9:
            self.health = 9

    def to_string(self) -> str:
        """Text representation of this unit."""
        p = self.player.name.lower()[0]
        t = self.type.name.upper()[0]
        return f"{p}{t}{self.health}"

    def __str__(self) -> str:
        """Text representation of this unit."""
        return self.to_string()

    def damage_amount(self, target: Unit) -> int:
        """How much can this unit damage another unit."""
        amount = self.damage_table[self.type.value][target.type.value]
        if target.health - amount < 0:
            return target.health
        return amount

    def repair_amount(self, target: Unit) -> int:
        """How much can this unit repair another unit."""
        amount = self.repair_table[self.type.value][target.type.value]
        if target.health + amount > 9:
            return 9 - target.health
        return amount


##############################################################################################################


@dataclass(slots=True)
class Coord:
    """Representation of a game cell coordinate (row, col)."""

    row: int = 0
    col: int = 0

    def __eq__(self, other: Coord) -> bool:
        # Comparing two coord objects (overloaded == opearator for the Coords)
        if self.row == other.row and self.col == other.col:
            return True
        else:
            return False

    def col_string(self) -> str:
        """Text representation of this Coord's column."""
        coord_char = "?"
        if self.col < 16:
            coord_char = "0123456789abcdef"[self.col]
        return str(coord_char)

    def row_string(self) -> str:
        """Text representation of this Coord's row."""
        coord_char = "?"
        if self.row < 26:
            coord_char = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[self.row]
        return str(coord_char)

    def to_string(self) -> str:
        """Text representation of this Coord."""
        return self.row_string() + self.col_string()

    def __str__(self) -> str:
        """Text representation of this Coord."""
        return self.to_string()

    def clone(self) -> Coord:
        """Clone a Coord."""
        return copy.copy(self)

    def iter_range(self, dist: int) -> Iterable[Coord]:
        """Iterates over Coords inside a rectangle centered on our Coord."""
        for row in range(self.row - dist, self.row + 1 + dist):
            for col in range(self.col - dist, self.col + 1 + dist):
                yield Coord(row, col)

    def iter_adjacent(self) -> Iterable[Coord]:
        """Iterates over adjacent Coords."""
        yield Coord(self.row - 1, self.col)
        yield Coord(self.row, self.col - 1)
        yield Coord(self.row + 1, self.col)
        yield Coord(self.row, self.col + 1)

    @classmethod
    def from_string(cls, s: str) -> Coord | None:
        """Create a Coord from a string. ex: D2."""
        s = s.strip()
        for sep in " ,.:;-_":
            s = s.replace(sep, "")
        if len(s) == 2:
            coord = Coord()
            coord.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[0:1].upper())
            coord.col = "0123456789abcdef".find(s[1:2].lower())
            return coord
        else:
            return None


##############################################################################################################


@dataclass(slots=True)
class CoordPair:
    """Representation of a game move or a rectangular area via 2 Coords."""

    src: Coord = field(default_factory=Coord)
    dst: Coord = field(default_factory=Coord)

    def to_string(self) -> str:
        """Text representation of a CoordPair."""
        return self.src.to_string() + " " + self.dst.to_string()

    def __str__(self) -> str:
        """Text representation of a CoordPair."""
        return self.to_string()

    def clone(self) -> CoordPair:
        """Clones a CoordPair."""
        return copy.copy(self)

    def iter_rectangle(self) -> Iterable[Coord]:
        """Iterates over cells of a rectangular area."""
        for row in range(self.src.row, self.dst.row + 1):
            for col in range(self.src.col, self.dst.col + 1):
                yield Coord(row, col)

    @classmethod
    def from_quad(cls, row0: int, col0: int, row1: int, col1: int) -> CoordPair:
        """Create a CoordPair from 4 integers."""
        return CoordPair(Coord(row0, col0), Coord(row1, col1))

    @classmethod
    def from_dim(cls, dim: int) -> CoordPair:
        """Create a CoordPair based on a dim-sized rectangle."""
        return CoordPair(Coord(0, 0), Coord(dim - 1, dim - 1))

    @classmethod
    def from_string(cls, s: str) -> CoordPair | None:
        """Create a CoordPair from a string. ex: A3 B2"""
        s = s.strip()
        for sep in " ,.:;-_":
            s = s.replace(sep, "")
        if len(s) == 4:
            coords = CoordPair()
            coords.src.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[0:1].upper())
            coords.src.col = "0123456789abcdef".find(s[1:2].lower())
            coords.dst.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[2:3].upper())
            coords.dst.col = "0123456789abcdef".find(s[3:4].lower())
            return coords
        else:
            return None


##############################################################################################################


@dataclass(slots=True)
class Options:
    """Representation of the game options."""

    dim: int = 5
    max_depth: int | None = 4
    min_depth: int | None = 2
    max_time: float | None = 5.0
    game_type: GameType = GameType.AttackerVsDefender
    alpha_beta: bool = True
    max_turns: int | None = 100
    randomize_moves: bool = True
    broker: str | None = None


##############################################################################################################


@dataclass(slots=True)
class Stats:
    """Representation of the global game statistics."""

    evaluations_per_depth: dict[int, int] = field(default_factory=dict)
    total_seconds: float = 0.0


##############################################################################################################


@dataclass(slots=True)
class Game:
    """Representation of the game state."""

    board: list[list[Unit | None]] = field(default_factory=list)
    next_player: Player = Player.Attacker
    turns_played: int = 0
    options: Options = field(default_factory=Options)
    stats: Stats = field(default_factory=Stats)
    _attacker_has_ai: bool = True
    _defender_has_ai: bool = True

    def __post_init__(self):
        """Automatically called after class init to set up the default board state."""
        dim = self.options.dim
        self.board = [[None for _ in range(dim)] for _ in range(dim)]
        md = dim - 1
        self.set(Coord(0, 0), Unit(player=Player.Defender, type=UnitType.AI))
        self.set(Coord(1, 0), Unit(player=Player.Defender, type=UnitType.Tech))
        self.set(Coord(0, 1), Unit(player=Player.Defender, type=UnitType.Tech))
        self.set(Coord(2, 0), Unit(player=Player.Defender, type=UnitType.Firewall))
        self.set(Coord(0, 2), Unit(player=Player.Defender, type=UnitType.Firewall))
        self.set(Coord(1, 1), Unit(player=Player.Defender, type=UnitType.Program))
        self.set(Coord(md, md), Unit(player=Player.Attacker, type=UnitType.AI))
        self.set(Coord(md - 1, md), Unit(player=Player.Attacker, type=UnitType.Virus))
        self.set(Coord(md, md - 1), Unit(player=Player.Attacker, type=UnitType.Virus))
        self.set(Coord(md - 2, md), Unit(player=Player.Attacker, type=UnitType.Program))
        self.set(Coord(md, md - 2), Unit(player=Player.Attacker, type=UnitType.Program))
        self.set(
            Coord(md - 1, md - 1), Unit(player=Player.Attacker, type=UnitType.Firewall)
        )

    def clone(self) -> Game:
        """Make a new copy of a game for minimax recursion.

        Shallow copy of everything except the board (options and stats are shared).
        """
        new = copy.copy(self)
        new.board = copy.deepcopy(self.board)
        return new

    def is_empty(self, coord: Coord) -> bool:
        """Check if contents of a board cell of the game at Coord is empty (must be valid coord)."""
        return self.board[coord.row][coord.col] is None

    def get(self, coord: Coord) -> Unit | None:
        """Get contents of a board cell of the game at Coord."""
        if self.is_valid_coord(coord):
            return self.board[coord.row][coord.col]
        else:
            return None

    def set(self, coord: Coord, unit: Unit | None):
        """Set contents of a board cell of the game at Coord."""
        if self.is_valid_coord(coord):
            self.board[coord.row][coord.col] = unit

    def remove_dead(self, coord: Coord):
        """Remove unit at Coord if dead."""
        unit = self.get(coord)
        if unit is not None and not unit.is_alive():
            self.set(coord, None)
            if unit.type == UnitType.AI:
                if unit.player == Player.Attacker:
                    self._attacker_has_ai = False
                else:
                    self._defender_has_ai = False

    def mod_health(self, coord: Coord, health_delta: int):
        """Modify health of unit at Coord (positive or negative delta)."""
        target = self.get(coord)
        if target is not None:
            target.mod_health(health_delta)
            self.remove_dead(coord)

    def check_combat(self, coord: Coord) -> bool:
        """Check if the piece is in a combat position"""
        # check for all adj positions
        for adj in coord.iter_adjacent():
            # retrieve that position
            unit = self.get(adj)
            # if piece is not empty and not ours, combat
            if unit is not None and unit.player != self.next_player:
                return True
        # if not no combat
        return False

    def combat_sequence(
        self, target: Unit, src: Unit, targ_coord: Coord, source_coord: Coord
    ):
        dmgInfl = src.damage_amount(target)
        dmgTaken = target.damage_amount(src)

        target.mod_health(-dmgInfl)
        src.mod_health(-dmgTaken)

        if target.health <= 0:
            self.remove_dead(targ_coord)

        if src.health <= 0:
            self.remove_dead(source_coord)

    def is_valid_advance(self, coords: CoordPair) -> bool:

        unit_src = self.get(coords.src)

        if not self.is_valid_coord(coords.src) or not self.is_valid_coord(coords.dst):
            return False
        
        if coords.dst not in coords.src.iter_adjacent():
            return False

        if (
            self.next_player == Player.Attacker
            and not self.validate_atacker_move(unit_src, coords.dst, coords.src)
        ) or (
            self.next_player == Player.Defender
            and not self.validate_defender_move(unit_src, coords.dst, coords.src)
        ):
            return False

        unit_dest = self.get(coords.dst)
        return unit_dest is None

    # validate atacker piece movement
    def validate_atacker_move(self, unit, dest, src):
        if unit.type in [UnitType.Program, UnitType.Firewall, UnitType.AI]:
            if dest in [Coord(src.row - 1, src.col), Coord(src.row, src.col - 1), Coord(src.row, src.col)]:
                return True
            return False
        else:
            return True

    # validate defender piece movement
    def validate_defender_move(self, unit, dest, src):
        if unit.type in [UnitType.Program, UnitType.Firewall, UnitType.AI]:
            if dest in [Coord(src.row + 1, src.col), Coord(src.row, src.col + 1), Coord(src.row, src.col)]:
                return True
            return False
        else:
            return True

    def get_sourounding_units(self, coord: Coord):
        units = []
        # Iterate over sourounding entities (including diagonals)
        for enplacement in coord.iter_range(1):
            units.append(enplacement)
        return units

    def self_destruct(self, src: Coord):
        # Retrieve all sourounding units
        sourounding_coords = self.get_sourounding_units(src)
        suicide_unit = self.get(src)

        if suicide_unit is None:
            return

        # iterate over the sourounding entities and damage their health (if unit exists)
        #  - if health smaller than or eqaul to zero, remove dead unit from board
        for place in sourounding_coords:
            entity = self.get(place)

            if entity is not None:
                entity.mod_health(-2)

                if entity.health <= 0:
                    self.remove_dead(src)

        # Remove the self destructed unit    
        suicide_unit.health = 0
        self.remove_dead(src)

    def validate_move(self, coords: CoordPair):
        unit_src = self.get(coords.src)
        target = self.get(coords.dst)

        if unit_src is None or unit_src.player != self.next_player:
            return MoveType.Invalid
        
        if coords.src == coords.dst:
            return MoveType.SelfDestruct
        
        elif (
            target is not None
            and unit_src.type in [UnitType.Tech, UnitType.AI]
            and target.player == self.next_player and unit_src.repair_amount(target) !=0
        ):
            return MoveType.Repair
        
        elif self.check_combat(coords.src):
            if target is not None and target.player != self.next_player:               
                return MoveType.Attack
            
            elif unit_src.type not in [UnitType.Tech, UnitType.Virus]:
                return MoveType.Invalid
        
        if self.is_valid_advance(coords):
            return MoveType.Advance
            
        return MoveType.Invalid
    

    def repair_friendly(
        self, target: Unit, src: Unit, targ_coord: Coord, source_coord: Coord
    ):
        hp_gained = src.repair_amount(target)
        target.mod_health(hp_gained)


    def perform_move(self, coords: CoordPair, file) -> Tuple[bool, str]:
        unit_src = self.get(coords.src)
        target = self.get(coords.dst)

        move_type = self.validate_move(coords)

        if(move_type is MoveType.Invalid):
            return (False, "invalid move")
        
        if(move_type is MoveType.Repair):
            self.repair_friendly(target, unit_src, coords.dst, coords.src)
            if(file is not None):
                file.write(
                    f"Move from {coords.src} to {coords.dst} - repair unit {target.to_string()}\n"
                )
            return (True, "")
        
        if(move_type is MoveType.Attack):
            self.combat_sequence(target, unit_src, coords.dst, coords.src)
            if(file is not None):
                file.write(
                    f"Move from {coords.src} to {coords.dst} - {unit_src.to_string()} attacks {target.to_string()}\n"
                )
            return (True, "")
        
        if(move_type is MoveType.Advance):
            self.set(coords.dst, self.get(coords.src))
            self.set(coords.src, None)
            if(file is not None):
                file.write(f"Move from {coords.src} to {coords.dst}\n")
            return (True, "")
        
        if (move_type is MoveType.SelfDestruct):
            if coords.src == coords.dst:
                self.self_destruct(coords.src)
                if(file is not None):
                    file.write(f"Move from {coords.src} to {coords.dst} - self destruct\n")
                return (True, "")
            

    def next_turn(self):
        """Transitions game to the next turn."""
        self.next_player = self.next_player.next()
        self.turns_played += 1

    def to_string(self) -> str:
        """Pretty text representation of the game."""
        dim = self.options.dim
        output = ""
        output += f"Next player: {self.next_player.name}\n"
        output += f"Turns played: {self.turns_played}\n"
        coord = Coord()
        output += "\n   "
        for col in range(dim):
            coord.col = col
            label = coord.col_string()
            output += f"{label:^3} "
        output += "\n"
        for row in range(dim):
            coord.row = row
            label = coord.row_string()
            output += f"{label}: "
            for col in range(dim):
                coord.col = col
                unit = self.get(coord)
                if unit is None:
                    output += " .  "
                else:
                    output += f"{str(unit):^3} "
            output += "\n"
        return output

    def __str__(self) -> str:
        """Default string representation of a game."""
        return self.to_string()

    def is_valid_coord(self, coord: Coord) -> bool:
        """Check if a Coord is valid within out board dimensions."""
        dim = self.options.dim
        if coord.row < 0 or coord.row >= dim or coord.col < 0 or coord.col >= dim:
            return False
        return True

    def read_move(self) -> CoordPair:
        """Read a move from keyboard and return as a CoordPair."""
        while True:
            s = input(f"Player {self.next_player.name}, enter your move: ")
            coords = CoordPair.from_string(s)
            if (
                coords is not None
                and self.is_valid_coord(coords.src)
                and self.is_valid_coord(coords.dst)
            ):
                return coords
            else:
                print("Invalid coordinates! Try again.")

    def human_turn(self, file):
        """Human player plays a move (or get via broker)."""
        if self.options.broker is not None:
            print("Getting next move with auto-retry from game broker...")
            while True:
                mv = self.get_move_from_broker()
                if mv is not None:
                    (success, result) = self.perform_move(mv, file)
                    print(f"Broker {self.next_player.name}: ", end="")
                    print(result)
                    if success:
                        self.next_turn()
                        break
                sleep(0.1)
        else:
            while True:
                mv = self.read_move()
                (success, result) = self.perform_move(mv, file)
                if success:
                    print(f"Player {self.next_player.name}: ", end="")
                    print(result)
                    self.next_turn()
                    break
                else:
                    print("The move is not valid! Try again.")

    def computer_turn(self, file) -> CoordPair | None:
        """Computer plays a move."""
        mv = self.suggest_move()
        if mv is not None:
            (success, result) = self.perform_move(mv, file)
            if success:
                print(f"Computer {self.next_player.name}: ", end="")
                print(result)
                self.next_turn()
        return mv

    def player_units(self, player: Player) -> Iterable[Tuple[Coord, Unit]]:
        """Iterates over all units belonging to a player."""
        for coord in CoordPair.from_dim(self.options.dim).iter_rectangle():
            unit = self.get(coord)
            if unit is not None and unit.player == player:
                yield (coord, unit)

    def is_finished(self) -> bool:
        """Check if the game is over."""
        return self.has_winner() is not None

    def has_winner(self) -> Player | None:
        """Check if the game is over and returns winner"""
        if self.options.max_turns is not None and self.turns_played >= self.options.max_turns:
            return Player.Defender
        if self._attacker_has_ai:
            if self._defender_has_ai:
                return None
            else:
                return Player.Attacker    
        return Player.Defender

    def move_candidates(self) -> Iterable[CoordPair]:
        """Generate valid move candidates for the next player."""
        move = CoordPair()
        for src, _ in self.player_units(self.next_player):
            move.src = src
            piece = self.get(src)

            for dst in src.iter_adjacent():
                move.dst = dst
                
                move_type = self.validate_move(move)

                if piece.type == UnitType.AI and move_type == MoveType.SelfDestruct:
                    continue

                if move_type is not MoveType.Invalid:
                    yield move.clone()

            move.dst = src

            # Do not allow the AI to self destruct
            if piece.type == UnitType.AI:
                continue

            yield move.clone()

        
    def minimax(self, depth: int, is_maxiPlayer: bool) -> (int, CoordPair, int):
        if depth == 0:
            # OG e0() heuristic
            # return e0_heuristic(self), None, depth
            # modified e0() to test health
            # return e0_heuristic_with_health(self), None, depth


            if is_maxiPlayer:
                return e1_attacker_heuristic(self), None, depth
        
            if not is_maxiPlayer:
                return e0_heuristic(self), None, depth
        
        possible_moves = self.move_candidates()

        if is_maxiPlayer:
            max_eval = MIN_HEURISTIC_SCORE
            optimal_move = None

            for move in possible_moves:
                new_game = self.clone()
                new_game.perform_move(move, None)
                new_game.next_player = Player.Defender
                (eval, move_performed, depth_stat) = new_game.minimax(depth - 1, False)

                if max_eval <= eval:
                    max_eval = eval
                    optimal_move = move

            return max_eval, optimal_move, 0
        
        else:
            min_eval = MAX_HEURISTIC_SCORE
            optimal_move = None

            for move in possible_moves:
                new_game = self.clone()
                new_game.perform_move(move, None)
                new_game.next_player = Player.Attacker
                (eval, move_performed, depth_stat) = new_game.minimax(depth - 1, True)

                if min_eval >= eval:
                    min_eval = eval
                    optimal_move = move

            return min_eval, optimal_move, 0
        

    def minimax_alpha_beta(self, depth: int, is_maxiPlayer: bool, alpha: int, beta :int) -> (int, CoordPair, int):
        if depth == 0 or self.is_finished():

            # OG e0() heuristic
            # return e0_heuristic(self), None, depth
            # modified e0() to test health
            # return e0_heuristic_with_health(self), None, depth

            if is_maxiPlayer:
                return e1_attacker_heuristic(self), None, depth
        
            if not is_maxiPlayer:
                return e0_heuristic(self), None, depth
            
            
        possible_moves = self.move_candidates()

        if is_maxiPlayer:
            max_eval = MIN_HEURISTIC_SCORE
            optimal_move = None

            for move in possible_moves:

                new_game = self.clone()

                # perform the new virtual move - to compute possible move
                new_game.perform_move(move, None)

                # change the turn of the virtual game for the next iteration - same is done for the defender (minimizing player)
                new_game.next_player = Player.Defender
                (state_heuristic, move_performed, depth_stat) = new_game.minimax_alpha_beta(depth - 1, False, alpha, beta)

                # Pruning
                alpha = max(alpha, state_heuristic)
                if beta <= alpha:
                    break

                if max_eval <= state_heuristic:
                    max_eval = state_heuristic
                    optimal_move = move

            return max_eval, optimal_move, 0
        
        else:
            min_eval = MAX_HEURISTIC_SCORE
            optimal_move = None

            for move in possible_moves:                

                new_game = self.clone()
                new_game.perform_move(move, None)

                new_game.next_player = Player.Attacker
                (state_heuristic, move_performed, depth_stat) = new_game.minimax_alpha_beta(depth - 1, True, alpha, beta)

                # Pruning
                beta = min(beta, state_heuristic)
                if beta <= alpha:
                    break

                if min_eval >= state_heuristic:
                    min_eval = state_heuristic
                    optimal_move = move

            return min_eval, optimal_move, 0


    def suggest_move(self) -> CoordPair | None:
        """Suggest the next move using minimax alpha beta. TODO: REPLACE RANDOM_MOVE WITH PROPER GAME LOGIC!!!"""
        start_time = datetime.now()

        # (score, move, avg_depth) = self.minimax_alpha_beta(10, True, MIN_HEURISTIC_SCORE, MAX_HEURISTIC_SCORE)
        if self.next_player == Player.Attacker:
            (score, move, avg_depth) = self.minimax_alpha_beta(10, True, MIN_HEURISTIC_SCORE, MAX_HEURISTIC_SCORE)
        else:
            (score, move, avg_depth) = self.minimax_alpha_beta(10, False, MIN_HEURISTIC_SCORE, MAX_HEURISTIC_SCORE)
            
        elapsed_seconds = (datetime.now() - start_time).total_seconds()
        self.stats.total_seconds += elapsed_seconds
        print(f"Heuristic score: {score}")
        print(f"Average recursive depth: {avg_depth:0.1f}")
        print(f"Evals per depth: ", end="")
        for k in sorted(self.stats.evaluations_per_depth.keys()):
            print(f"{k}:{self.stats.evaluations_per_depth[k]} ", end="")
        print()
        total_evals = sum(self.stats.evaluations_per_depth.values())
        if self.stats.total_seconds > 0:
            print(f"Eval perf.: {total_evals/self.stats.total_seconds/1000:0.1f}k/s")
        print(f"Elapsed time: {elapsed_seconds:0.1f}s")
        return move

    def post_move_to_broker(self, move: CoordPair):
        """Send a move to the game broker."""
        if self.options.broker is None:
            return
        data = {
            "from": {"row": move.src.row, "col": move.src.col},
            "to": {"row": move.dst.row, "col": move.dst.col},
            "turn": self.turns_played,
        }
        try:
            r = requests.post(self.options.broker, json=data)
            if (
                r.status_code == 200
                and r.json()["success"]
                and r.json()["data"] == data
            ):
                # print(f"Sent move to broker: {move}")
                pass
            else:
                print(
                    f"Broker error: status code: {r.status_code}, response: {r.json()}"
                )
        except Exception as error:
            print(f"Broker error: {error}")

    def get_move_from_broker(self) -> CoordPair | None:
        """Get a move from the game broker."""
        if self.options.broker is None:
            return None
        headers = {"Accept": "application/json"}
        try:
            r = requests.get(self.options.broker, headers=headers)
            if r.status_code == 200 and r.json()["success"]:
                data = r.json()["data"]
                if data is not None:
                    if data["turn"] == self.turns_played + 1:
                        move = CoordPair(
                            Coord(data["from"]["row"], data["from"]["col"]),
                            Coord(data["to"]["row"], data["to"]["col"]),
                        )
                        print(f"Got move from broker: {move}")
                        return move
                    else:
                        # print("Got broker data for wrong turn.")
                        # print(f"Wanted {self.turns_played+1}, got {data['turn']}")
                        pass
                else:
                    # print("Got no data from broker")
                    pass
            else:
                print(
                    f"Broker error: status code: {r.status_code}, response: {r.json()}"
                )
        except Exception as error:
            print(f"Broker error: {error}")
        return None

##############################################################################################################


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(
        prog="ai_wargame", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--max_depth", type=int, help="maximum search depth")
    parser.add_argument("--max_time", type=float, help="maximum search time")
    parser.add_argument(
        "--game_type",
        type=str,
        default="manual",
        help="game type: auto|attacker|defender|manual",
    )
    parser.add_argument("--broker", type=str, help="play via a game broker")
    parser.add_argument("--max_turns", type=int, help="maximum turns")
    args = parser.parse_args()

    # parse the game type
    if args.game_type == "attacker":
        game_type = GameType.AttackerVsComp
    elif args.game_type == "defender":
        game_type = GameType.CompVsDefender
    elif args.game_type == "manual":
        # TODO change at end to manual
        game_type = GameType.CompVsComp 
    else:
        game_type = GameType.CompVsComp

    # set up game options
    options = Options(game_type=game_type)

    # override class defaults via command line options
    if args.max_depth is not None:
        options.max_depth = args.max_depth
    if args.max_time is not None:
        options.max_time = args.max_time
    if args.broker is not None:
        options.broker = args.broker
    if args.max_turns is not None:
        options.max_turns = args.max_turns

    # create a new game
    game = Game(options=options)

    filename = f"gameTrace-{str(options.alpha_beta).lower()}-{int(options.max_time)}-{int(options.max_turns)}.txt"

    f = open(filename, "w")
    f.write(f"Timeout is {options.max_time} \n")
    f.write(f"Max turns is {options.max_turns} \n")
    if game._attacker_has_ai or game._defender_has_ai:
        f.write(f"Alpha-beta is {options.alpha_beta} \n")
    if game._attacker_has_ai:
        f.write(f"Player 1 = AI \n")
    else:
        f.write(f"Player 1 is H \n")
    if game._defender_has_ai:
        f.write(f"Player 2 = AI \n")
    else:
        f.write(f"Player 2 is H \n")
    # name heuristic???
    f.write(f"{game} \n")
    f.write(
        "------------------------------------------------------------------------------"
    )

    # the main game loop
    while True:
        print()
        print(game)
        winner = game.has_winner()
        f.write(f"\nTurn #{game.turns_played} \n")
        f.write(f"Player {game.next_player.name} \n")
        if winner is not None:
            print(f"{winner.name} wins!")
            # writes number of turns played for winner
            f.write(f"{winner.name} wins in {game.turns_played} turns!")
            break
        if game.options.game_type == GameType.AttackerVsDefender:
            game.human_turn(f)
        elif (
            game.options.game_type == GameType.AttackerVsComp
            and game.next_player == Player.Attacker
        ):
            game.human_turn(f)
        elif (
            game.options.game_type == GameType.CompVsDefender
            and game.next_player == Player.Defender
        ):
            game.human_turn(f)
        else:
            player = game.next_player
            move = game.computer_turn(f)
            if move is not None:
                game.post_move_to_broker(move)
            else:
                print("Computer doesn't know what to do!!!")
                exit(1)

        # if ai action time?
        # if ai heuristic score?
        f.write("New configuration of the board:\n")
        f.write(f"{game}\n")


##############################################################################################################

def count_pieces_by_player(game: Game):

    total_health_attacker = 0
    total_health_defender = 0

    piece_count = {
        Player.Attacker: {
            UnitType.Virus: 0,
            UnitType.Tech: 0,
            UnitType.Firewall: 0,
            UnitType.Program: 0,
            UnitType.AI: 0,
        },
        Player.Defender: {
            UnitType.Virus: 0,
            UnitType.Tech: 0,
            UnitType.Firewall: 0,
            UnitType.Program: 0,
            UnitType.AI: 0,
        },
    }

    for row in game.board:
        for piece in row:

            if piece:
                piece_count[piece.player][piece.type] += 1
                
                if piece.player == Player.Attacker:
                    total_health_attacker += piece.health
                else:
                    total_health_defender += piece.health   

    return piece_count, total_health_attacker, total_health_defender

# Assuming P1 is the attacker - heuristic from the handout
def e0_heuristic(game: Game) -> int:
    (dict_pieces, health_attack, health_defend) = count_pieces_by_player(game)
    attacker_sum = 3 * dict_pieces[Player.Attacker][UnitType.Virus] + 3 * dict_pieces[Player.Attacker][UnitType.Tech] + 3 * dict_pieces[Player.Attacker][UnitType.Firewall] + 3 * dict_pieces[Player.Attacker][UnitType.Program] + 9999 * dict_pieces[Player.Attacker][UnitType.AI]
    defender_sum = 3 * dict_pieces[Player.Defender][UnitType.Virus] + 3 * dict_pieces[Player.Defender][UnitType.Tech] + 3 * dict_pieces[Player.Defender][UnitType.Firewall] + 3 * dict_pieces[Player.Defender][UnitType.Program] + 9999 * dict_pieces[Player.Defender][UnitType.AI]
    return (attacker_sum - defender_sum)


def e0_heuristic_with_health(game: Game) -> int:
    (dict_pieces, health_attack, health_defend) = count_pieces_by_player(game)
    return (health_attack - health_defend)


def attack_power(dict_pieces, health_attack, health_defend) -> int:

    virus_att_pwr = dict_pieces[Player.Attacker][UnitType.Virus] * (9 * dict_pieces[Player.Defender][UnitType.AI] + 3 * dict_pieces[Player.Defender][UnitType.Tech] + 1 * dict_pieces[Player.Defender][UnitType.Firewall] + 6 * dict_pieces[Player.Defender][UnitType.Program])
    firewall_att_pwr = dict_pieces[Player.Attacker][UnitType.Firewall] * (dict_pieces[Player.Defender][UnitType.AI] +  dict_pieces[Player.Defender][UnitType.Tech] + dict_pieces[Player.Defender][UnitType.Firewall] + dict_pieces[Player.Defender][UnitType.Program])
    program_att_pwr = dict_pieces[Player.Attacker][UnitType.Program] * (3 * dict_pieces[Player.Defender][UnitType.AI] + 3 * dict_pieces[Player.Defender][UnitType.Tech] + 1 * dict_pieces[Player.Defender][UnitType.Firewall] + 3 * dict_pieces[Player.Defender][UnitType.Program])
    ai_att_pwr = dict_pieces[Player.Attacker][UnitType.AI] * (3 * dict_pieces[Player.Defender][UnitType.AI] + 3 * dict_pieces[Player.Defender][UnitType.Tech] + 1 * dict_pieces[Player.Defender][UnitType.Firewall] + 3 * dict_pieces[Player.Defender][UnitType.Program])

    tech_def_pwr = dict_pieces[Player.Defender][UnitType.Tech] * (1 * dict_pieces[Player.Attacker][UnitType.AI] + 6 * dict_pieces[Player.Attacker][UnitType.Virus] + 1 * dict_pieces[Player.Attacker][UnitType.Firewall] + 1 * dict_pieces[Player.Attacker][UnitType.Program])
    firewall_def_pwr = dict_pieces[Player.Defender][UnitType.Firewall] * (dict_pieces[Player.Attacker][UnitType.AI] +  dict_pieces[Player.Attacker][UnitType.Virus] + dict_pieces[Player.Attacker][UnitType.Firewall] + dict_pieces[Player.Attacker][UnitType.Program])
    program_def_pwr = dict_pieces[Player.Defender][UnitType.Program] * (3 * dict_pieces[Player.Attacker][UnitType.AI] + 3 * dict_pieces[Player.Attacker][UnitType.Virus] + 1 * dict_pieces[Player.Attacker][UnitType.Firewall] + 3 * dict_pieces[Player.Attacker][UnitType.Program])
    ai_def_pwr = dict_pieces[Player.Defender][UnitType.AI] * (3 * dict_pieces[Player.Attacker][UnitType.AI] + 3 * dict_pieces[Player.Attacker][UnitType.Virus] + 1 * dict_pieces[Player.Attacker][UnitType.Firewall] + 3 * dict_pieces[Player.Attacker][UnitType.Program])

    attacker_stats = health_attack + virus_att_pwr + firewall_att_pwr + program_att_pwr + ai_att_pwr
    defender_stats = health_defend + tech_def_pwr + firewall_def_pwr + program_def_pwr + ai_def_pwr

    return (attacker_stats - defender_stats)
   

def find_optimal_oponent(attacking_piece: Unit, piece_count):
    opponents_damages = attacking_piece.damage_table[attacking_piece.type.value]
    max = 0
    max_opp_index = 0

    enemy_type = Player.Defender if attacking_piece.player == Player.Attacker else Player.Attacker

    for opp, value in enumerate(opponents_damages):
        if piece_count[enemy_type][UnitType(opp)] == 0:
            continue
        if value > max:
            max = value
            max_opp_index = opp

    return UnitType(max_opp_index), max

def BFS(game: Game, target: UnitType, src: Coord):
    visited = []
    queue = []

    queue.append(src)
    visited.append(src)

    while queue:
        node = queue.pop(0)
        node_val = game.get(node)

        if node_val and node_val.type == target and node_val.player != game.next_player:
            return node

        for adjecent in node.iter_adjacent():
            if adjecent not in visited and game.is_valid_coord(adjecent):
                visited.append(adjecent)
                queue.append(adjecent)
    

def calc_distance(src : Coord, target : Coord):


    src_x = src.row
    src_y = src.col

    targ_x = target.row
    targ_y = target.col

    answ = (((src_x - targ_x)**2 + (src_y - targ_y)**2)**0.5)

    return answ


# Attacker's heuristic 
def e1_attacker_heuristic(game: Game) -> int:

    piece_values = {
        UnitType.Virus: 8,
        UnitType.Tech: 8,
        UnitType.Firewall: 3,
        UnitType.Program: 5,
        UnitType.AI: 10,        
    }

    (piece_count, health_attack, health_defender) = count_pieces_by_player(game)

    if piece_count[Player.Attacker][UnitType.AI] == 0:
        return -9999
    
    if piece_count[Player.Defender][UnitType.AI] == 0:
        return 9999

    score = 0

    health_bonus = health_attack - health_defender
    attack_power_bonus = attack_power(piece_count, health_attack, health_defender)

    for row_num, row in enumerate(game.board):
        for col_num, piece in enumerate(row):
            if piece and piece.player == game.next_player:           

                    src_coord = Coord (row_num, col_num)

                    max_dammage_opp, dammage = find_optimal_oponent(piece, piece_count)
                    target_coord = BFS(game, max_dammage_opp, src_coord)

                    distance = calc_distance(src_coord, target_coord)

                    piece_heuristic = (dammage * (piece_values[max_dammage_opp] - distance)).__ceil__()

                    if game.next_player == Player.Defender:
                        piece_heuristic *= -1

                    score += piece_heuristic

    return (score + health_bonus + attack_power_bonus)   

if __name__ == "__main__":
    main()
