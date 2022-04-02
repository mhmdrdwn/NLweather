from src.train import train_speed_data, train_direction_data


def main():
    print('Speed Data...')
    train_speed_data()
    print('Direction Data')
    train_direction_data()
    
if __name__ == '__main__':
    main()
