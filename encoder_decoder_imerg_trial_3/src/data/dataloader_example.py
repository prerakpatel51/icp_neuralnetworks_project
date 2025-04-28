from data_provider import IMERGDataModule

# event id for which the data was downloaded
event_id = 'WA_dataset'

# location of the h5 file that was generated after downloading the data
# h5_dataset_location = '../data/events/'+str(event_id)+'.h5'
h5_dataset_location='/home1/ppatel2025/ppworktp/encoder_decoder_imerg_trial_1/src/data/dataset/WA_dataset.h5'
# as of now, we do not have IR data, so we set it None
ir_h5_dataset_location = None

# this string is used to determine the kind of dataloader we need to use
# for processing individual events, we reccommend the user to keep this fixed
dataset_type = 'wa_expanded'


data_provider =  IMERGDataModule(
        forecast_steps = 12,
        history_steps = 8,
        imerg_filename = h5_dataset_location,
        ir_filename = ir_h5_dataset_location,
        batch_size = 32,
        image_shape = (360, 516),
        normalize_data=False,
        dataset = dataset_type)

test_data_loader = data_provider.test_dataloader()
train_data_loader = data_provider.train_dataloader()
val_data_loader = data_provider.val_dataloader()


