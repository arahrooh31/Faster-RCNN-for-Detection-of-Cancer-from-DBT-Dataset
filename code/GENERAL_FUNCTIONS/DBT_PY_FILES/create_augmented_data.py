def create_augmented_data(image_data,flip = 1,rot90=1,rot180=1,rot270=1,patchx =244,patchy=244):
    #flip and rotate the data to make new data files

    import numpy as np
    
    output90 = np.zeros([3,patchx,patchy])
    output180 = np.zeros([3,patchx,patchy])
    output270 = np.zeros([3,patchx,patchy])
    outputflip = np.zeros([3,patchx,patchy])

    #print(np.shape(output90))

    for ii in range(0,np.shape(image_data)[0]):
        image90 = np.rot90(image_data[ii,:,:])
        image180 = np.rot90(image_data[ii,:,:],k=2)
        image270 = np.rot90(image_data[ii,:,:],k=3)
        imageflip = np.fliplr(image_data[ii,:,:])
        output90[ii,:,:] = image90
        output180[ii,:,:] = image180
        output270[ii,:,:] = image270
        outputflip[ii,:,:] = imageflip



    return output90, output180, output270, outputflip