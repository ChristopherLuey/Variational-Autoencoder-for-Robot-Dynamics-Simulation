import math
import torch
import gym
import numpy as np
import sqlite3
import json


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
    if hasattr(env, 'reset'):  # Check if the environment supports seeding directly
        env.reset(seed=seed)
    else:
        print("Warning: Environment does not support seeding in reset method.")
        env.reset()
    
    if control_sequence.numel() % joints != 0:
        raise ValueError("Control sequence length must be divisible by the number of joints.")
    
    control_sequence = control_sequence.view(-1, joints)
    if control_sequence.shape[1] != joints:
        raise ValueError(f"Control sequence reshaped incorrectly. Expected second dimension to be {joints}, but got {control_sequence.shape[1]}.")
    
    total_velocity_x, total_velocity_y = 0, 0
    total_reward = 0
    positions = []
    velocities = []
    directions = []
    rewards = []

    # Initialize previous_action outside the loop to avoid conditional logic
    previous_action = control_sequence[0].cpu().numpy() if control_sequence.shape[0] > 0 else None
    interpolated_actions = []

    for action in control_sequence.cpu().numpy():
        action = control_sequence[i].cpu().numpy()

        # Vectorize the interpolation step for efficiency
        if previous_action is not None:
            interpolated_actions = np.linspace(previous_action, action, substeps + 1, endpoint=False)[1:]
        else:
            interpolated_actions = [action]

        for interpolated_action in interpolated_actions:
            observation, reward, done, info = env.step(interpolated_action)
            current_position = {'x': info["x_position"], 'y': info["y_position"]}
            current_velocity = {'x': info["x_velocity"], 'y': info["y_velocity"]}
            current_direction = math.atan2(2.0 * (observation[3] * observation[0] + observation[1] * observation[2]), 1 - 2 * (observation[0] ** 2 + observation[2] ** 2)) / (2 * math.pi)

            # Record positions, velocities, directions, and rewards over time
            positions.append(current_position)
            velocities.append(current_velocity)
            directions.append(current_direction)
            rewards.append(reward)

            # Update total velocity and reward
            total_velocity_x += current_velocity['x']
            total_velocity_y += current_velocity['y']
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
        "x_positions_over_time": [pos['x'] for pos in positions],
        "y_positions_over_time": [pos['y'] for pos in positions],
        "x_positions_over_time": [vel['x'] for vel in velocities],
        "y_positions_over_time": [vel['x'] for vel in velocities],
        "directions_over_time": directions,
        "rewards_over_time": rewards
    }
    
    return final_info


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
                "final_position": row[7],
                "total_velocity": row[8],
                "direction": row[9],
                "positions_over_time": json.loads(row[10]),
                "velocities_over_time": json.loads(row[11]),
                "directions_over_time": json.loads(row[12]),
                "rewards_over_time": json.loads(row[13])
            })
        return results


if __name__ == "__main__":
    # Set up the environment and parameters for database creation
    env = gym.make('Ant-v3', terminate_when_unhealthy=True, healthy_z_range=(0.3, 5), ctrl_cost_weight=0, contact_cost_weight=0, healthy_reward=0)
    env.model.opt.timestep = 0.01  # Set timestep to 0.01 seconds for higher resolution control
    joints = 8  # Assuming 8 joints for the Ant environment; adjust as needed
    control_sequence_length = 100  # Length of control sequence
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Set up SQLite database connection
    conn = sqlite3.connect('control_sequences.db')
    cursor = conn.cursor()

    # Create table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS control_sequences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            seed INTEGER,
            temporal_resolution REAL,
            temporal_substeps INTEGER,
            temporal_run TEXT,
            joints INTEGER,
            control TEXT,
            final_position REAL,
            total_velocity REAL,
            direction REAL,
            positions_over_time TEXT,
            velocities_over_time TEXT,
            directions_over_time TEXT,
            rewards_over_time TEXT
        )
    ''')

    # Generate a large database of control sequences and their corresponding evaluations
    num_sequences = 10000  # Number of control sequences to generate for the database

    for seed in range(num_sequences):
        control_sequence = (2 * torch.rand(control_sequence_length * joints) - 1).to(device)
        final_info = evaluate_control_sequence(control_sequence, env, joints, seed=seed)

        # Insert into SQL database
        cursor.execute('''
            INSERT INTO control_sequences (seed, temporal_resolution, temporal_substeps, temporal_run, joints, control, final_position, total_velocity, direction, positions_over_time, velocities_over_time, directions_over_time, rewards_over_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            final_info['seed'],
            final_info['temporal_resolution'],
            final_info['temporal_substeps'],
            final_info['temporal_run'],
            final_info['joints'],
            json.dumps(control_sequence.view(-1, joints).cpu().tolist()),
            final_info['final_position'],
            final_info['total_velocity'],
            final_info['direction'],
            json.dumps(final_info['positions_over_time']),
            json.dumps(final_info['velocities_over_time']),
            json.dumps(final_info['directions_over_time']),
            json.dumps(final_info['rewards_over_time'])
        ))

        # Commit every 1000 records to avoid excessive IO operations
        if (seed + 1) % 1000 == 0:
            conn.commit()

    # Commit any remaining data after the final batch
    conn.commit()

    # Close the environment and database connection
    env.close()
    conn.close()

    # Example query to retrieve data based on a specific column value
    results = query_database("seed", 42)
    for result in results:
        print(result)