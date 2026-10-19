"""RoboDojo data components for Ego4WAM.

Training imports lerobot_datasets lazily from train_ego4wam.create_dataloader.
Importing the online policy must not initialize unrelated training readers.
"""
