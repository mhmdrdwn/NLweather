#!/usr/bin/env python3

__author__ = 'Mohamed Radwan'


from src.train import train_speed_data, train_direction_data


def main():
    num_epochs=10
    print('Speed Data...')
    train_speed_data(num_epochs)
    #print('Direction Data')
    #train_direction_data(num_epochs)
    
if __name__ == '__main__':
    main()
