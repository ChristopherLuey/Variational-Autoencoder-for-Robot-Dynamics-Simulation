import math
import torch
import gym
import numpy as np
import sqlite3
import json
import random
from tabulate import tabulate


# Function to create the database with the reward column
def create_database():
    with sqlite3.connect('control_sequences.db') as conn:
        cursor = conn.cursor()
        # Drop the existing table (WARNING: This will delete all existing data in the table)
        cursor.execute('DROP TABLE IF EXISTS control_sequences')

        # Create table with the updated schema including the reward column
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS control_sequences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                seed INTEGER,
                temporal_resolution REAL,
                temporal_substeps INTEGER,
                temporal_run TEXT,
                joints INTEGER,
                control TEXT,
                final_x_position REAL,
                final_y_position REAL,
                final_x_velocity REAL,
                final_y_velocity REAL,
                direction REAL,
                x_positions_over_time TEXT,
                y_positions_over_time TEXT,
                x_velocities_over_time TEXT,
                y_velocities_over_time TEXT,
                directions_over_time TEXT,
                rewards_over_time TEXT,
                reward REAL
            )
        ''')
        conn.commit()


# Function to add control sequences to the database
def add_control_sequences(num_sequences, env, joints, control_sequence_length, device):
    with sqlite3.connect('control_sequences.db') as conn:
        cursor = conn.cursor()
        for seed in range(num_sequences):
            control_sequence = (2 * torch.rand(control_sequence_length * joints) - 1).to(device)
            final_info = evaluate_control_sequence(control_sequence, env, joints, seed=seed)

            # Calculate the reward as the sum of rewards_over_time

            # Insert into SQL database
            cursor.execute('''
                INSERT INTO control_sequences (seed, temporal_resolution, temporal_substeps, temporal_run, joints, control, final_x_position, final_y_position, final_x_velocity, final_y_velocity, direction, x_positions_over_time, y_positions_over_time, x_velocities_over_time, y_velocities_over_time, directions_over_time, rewards_over_time, reward)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                final_info['seed'],
                final_info['temporal_resolution'],
                final_info['temporal_substeps'],
                final_info['temporal_run'],
                final_info['joints'],
                json.dumps(final_info['control']),
                final_info['final_x_position'],
                final_info['final_y_position'],
                final_info['final_x_velocity'],
                final_info['final_y_velocity'],
                final_info['direction'],
                json.dumps(final_info['x_positions_over_time']),
                json.dumps(final_info['y_positions_over_time']),
                json.dumps(final_info['x_velocities_over_time']),
                json.dumps(final_info['y_velocities_over_time']),
                json.dumps(final_info['directions_over_time']),
                json.dumps(final_info['rewards_over_time']),
                sum(final_info['rewards_over_time'])
            ))

            # Commit every 1000 records to avoid excessive IO operations
            if (seed + 1) % 1000 == 0:
                conn.commit()

        # Commit any remaining data after the final batch
        conn.commit()
    conn.close()


def evaluate_control_sequence(control_sequence, env, joints, substeps=10, seed=None):
    """
    Evaluate a control sequence and return final information including position, velocity, direction, and more detailed metrics.

    Parameters:
    - control_sequence (torch.Tensor): The control sequence to apply.
    - env (gym.Env): The environment to run the control sequence in.
    - joints (int): Number of joints in the control sequence.
    - substeps (int): Number of substeps for each control action.
    - seed (int, optional): Seed for environment reset to ensure reproducibility.

    Returns:
    - dict: A dictionary containing metrics such as final position, velocity, direction, positions over time, velocities over time, directions over time, and rewards over time.
    """
    env.reset()
    
    if control_sequence.numel() % joints != 0:
        raise ValueError("Control sequence length must be divisible by the number of joints.")
    
    control_sequence = control_sequence.view(-1, joints)
    if control_sequence.shape[1] != joints:
        raise ValueError(f"Control sequence reshaped incorrectly. Expected second dimension to be {joints}, but got {control_sequence.shape[1]}.")
    
    total_velocity_x, total_velocity_y = 0, 0
    total_reward = 0
    positions_x = []
    positions_y = []
    velocities_x = []
    velocities_y = []
    directions = []
    rewards = []

    # Initialize previous_action to zero at the start
    previous_action = np.zeros_like(control_sequence[0].cpu().numpy()) if control_sequence.shape[0] > 0 else np.zeros(joints)
    interpolated_actions = []

    for action in control_sequence.cpu().numpy():

        # Vectorize the interpolation step for efficiency
        interpolated_actions = np.linspace(previous_action, action, substeps + 1, endpoint=False)[1:]
        

        for interpolated_action in interpolated_actions:
            observation, reward, done, info = env.step(interpolated_action)
            current_direction = math.atan2(
                2.0 * (observation[3] * observation[0] + observation[1] * observation[2]),
                1 - 2 * (observation[0] ** 2 + observation[2] ** 2)
            ) / (2 * math.pi)

            # Record positions, velocities, directions, and rewards over time
            positions_x.append(info["x_position"])
            positions_y.append(info["y_position"])
            velocities_x.append(info["x_velocity"])
            velocities_y.append(info["y_velocity"])
            directions.append(current_direction)
            rewards.append(reward)

            # Update total velocity and reward
            total_velocity_x += info["x_velocity"]
            total_velocity_y += info["y_velocity"]
            total_reward += reward

            if done:
                # Handle 'done' gracefully by breaking the loop and not resetting immediately
                break

        if done:
            break

        previous_action = action  # Update the previous action

    final_info = {
        "temporal_resolution": 0.01,
        "temporal_substeps": 10,
        "temporal_run": "linear",
        "joints": joints,
        "control": control_sequence.cpu().numpy().tolist(),
        "seed": seed,
        "final_x_position": info["x_position"],
        "final_y_position": info["y_position"],
        "final_x_velocity": total_velocity_x,
        "final_y_velocity": total_velocity_y,
        "direction": current_direction,
        "x_positions_over_time": positions_x,
        "y_positions_over_time": positions_y,
        "x_velocities_over_time": velocities_x,  # Corrected key
        "y_velocities_over_time": velocities_y,  # Corrected key
        "directions_over_time": directions,
        "rewards_over_time": rewards,
        "reward": sum(reward)
    }
    
    return final_info


def query_sequences_near_position(x_target=0.1, y_target=0, x_tolerance=0.05, y_tolerance=0.05):
    """
    Query the SQLite database to find sequences where the final position is around the given x and y values.

    Parameters:
    - x_target (float): Target x position.
    - y_target (float): Target y position.
    - x_tolerance (float): Allowed tolerance for x position.
    - y_tolerance (float): Allowed tolerance for y position.

    Returns:
    - list: A list of dictionaries containing the sequences that meet the position criteria.
    """
    with sqlite3.connect('control_sequences.db') as conn:
        cursor = conn.cursor()
        query = '''
        SELECT * FROM control_sequences
        WHERE final_x_position BETWEEN ? AND ?
        AND final_y_position BETWEEN ? AND ?
        '''
        cursor.execute(query, (
            x_target - x_tolerance, x_target + x_tolerance,
            y_target - y_tolerance, y_target + y_tolerance
        ))
        rows = cursor.fetchall()
        results = []
        for row in rows:
            results.append({
                "id": row[0],
                "seed": row[1],
                "temporal_resolution": row[2],
                "temporal_substeps": row[3],
                "temporal_run": row[4],
                "joints": row[5],
                "control": json.loads(row[6]),
                "final_x_position": row[7],
                "final_y_position": row[8],
                "final_x_velocity": row[9],
                "final_y_velocity": row[10],
                "direction": row[11],
                "x_positions_over_time": json.loads(row[12]),
                "y_positions_over_time": json.loads(row[13]),
                "x_velocities_over_time": json.loads(row[14]),
                "y_velocities_over_time": json.loads(row[15]),
                "directions_over_time": json.loads(row[16]),
                "rewards_over_time": json.loads(row[17]),
                "reward": row[18]
            })
    conn.close()
    return results


def query_sequences_near_reward(reward=1.0, tol=0.05):
    """
    Query the SQLite database to find sequences where the final position is around the given x and y values.

    Parameters:
    - reward (float): Target reward

    Returns:
    - list: A list of dictionaries containing the sequences that meet the position criteria.
    """
    with sqlite3.connect('control_sequences.db') as conn:
        cursor = conn.cursor()
        query = '''
        SELECT * FROM control_sequences
        WHERE reward BETWEEN ? AND ?
        '''
        cursor.execute(query, (reward-tol, reward+tol))
        rows = cursor.fetchall()
        results = []
        for row in rows:
            results.append({
                "id": row[0],
                "seed": row[1],
                "temporal_resolution": row[2],
                "temporal_substeps": row[3],
                "temporal_run": row[4],
                "joints": row[5],
                "control": json.loads(row[6]),
                "final_x_position": row[7],
                "final_y_position": row[8],
                "final_x_velocity": row[9],
                "final_y_velocity": row[10],
                "direction": row[11],
                "x_positions_over_time": json.loads(row[12]),
                "y_positions_over_time": json.loads(row[13]),
                "x_velocities_over_time": json.loads(row[14]),
                "y_velocities_over_time": json.loads(row[15]),
                "directions_over_time": json.loads(row[16]),
                "rewards_over_time": json.loads(row[17]),
                "reward": row[18]
            })
            # # Start of Selection
            # print(len(results))
            # print(type(results))
            # print(type(results[0]))
            # print(tabulate(results, headers="keys", tablefmt="grid"))
            return results


def query_database(column, value):
    """
    Query the SQLite database based on the provided column and value.

    Parameters:
    - column (str): The column name to filter the query.
    - value: The value to filter by.

    Returns:
    - list: A list of dictionaries containing the row data for the given column and value.
    """
    with sqlite3.connect('control_sequences.db') as conn:
        cursor = conn.cursor()
        query = f"SELECT * FROM control_sequences WHERE {column}=?"
        cursor.execute(query, (value,))
        rows = cursor.fetchall()
        results = []
        for row in rows:
            results.append({
                "id": row[0],
                "seed": row[1],
                "temporal_resolution": row[2],
                "temporal_substeps": row[3],
                "temporal_run": row[4],
                "joints": row[5],
                "control": json.loads(row[6]),
                "final_x_position": row[7],
                "final_y_position": row[8],
                "final_x_velocity": row[9],
                "final_y_velocity": row[10],
                "direction": row[11],
                "x_positions_over_time": json.loads(row[12]),
                "y_positions_over_time": json.loads(row[13]),
                "x_velocities_over_time": json.loads(row[14]),  # Corrected key
                "y_velocities_over_time": json.loads(row[15]),  # Corrected key
                "directions_over_time": json.loads(row[16]),
                "rewards_over_time": json.loads(row[17]),
                "reward": row[18]
            })
    conn.close()
    return results


# Function to add a 'reward' column to the table and populate it with the sum of rewards_over_time
def add_reward_column():
    with sqlite3.connect('control_sequences.db') as conn:
        cursor = conn.cursor()
        # Add 'reward' column to the table if it doesn't exist
        try:
            cursor.execute("ALTER TABLE control_sequences ADD COLUMN reward REAL")
        except sqlite3.OperationalError:
            # If the column already exists, skip adding it
            pass
        
        # Update the reward column for each row
        cursor.execute('SELECT id, rewards_over_time FROM control_sequences')
        rows = cursor.fetchall()
        for row in rows:
            row_id = row[0]
            rewards_over_time = json.loads(row[1])
            reward_sum = sum(rewards_over_time)
            cursor.execute('UPDATE control_sequences SET reward=? WHERE id=?', (reward_sum, row_id))
            print(row_id)
        # Commit the changes
        conn.commit()
    conn.close()


# Function to print a random row from the database for verification
def print_random_row():
    with sqlite3.connect('control_sequences.db') as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM control_sequences')
        row_count = cursor.fetchone()[0]
        if row_count == 0:
            print("No records found in the database.")
            return
        random_id = random.randint(1, row_count)
        cursor.execute('SELECT * FROM control_sequences WHERE id=?', (random_id,))
        row = cursor.fetchone()
        if row:
            print("Random Row:")
            print(row)
        else:
            print("No row found with the given ID.")
    conn.close()
    

if __name__ == "__main__":
    # Set up the environment and parameters for database creation
    env = gym.make('Ant-v3', terminate_when_unhealthy=True, healthy_z_range=(0.3, 5), ctrl_cost_weight=0, contact_cost_weight=0, healthy_reward=0)
    env.model.opt.timestep = 0.01  # Set timestep to 0.01 seconds for higher resolution control
    joints = 8  # Assuming 8 joints for the Ant environment; adjust as needed
    control_sequence_length = 100  # Length of control sequence
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Close the environment and database connection
    env.close()


    query_sequences_near_reward(10.0)
