def randomize_patches(islice, irows, icols,slice_selection, x_corner, y_corner, x_width,y_height):
    #take in the annotation area and move around that to get a new patch
    import numpy as np

    num_boxes = 10
    patch_x = 244
    patch_y = 244

    row_corner = y_corner
    col_corner = x_corner
    box_slice = slice_selection

    #Set the amount the random point can extend beyond the original patch area
    roi_delta = 0.1
    patch_size= 244
    patch_gap = patch_size * roi_delta


    #This is the allowed range of box centers, all within the given ROI
    #selectable_col = range(box_info[4],box_info[4]+box_info[5])
    col_limit = x_corner + x_width  #the column range in image (pixels)
    row_limit = y_corner + y_height #the row range in image (pixels)
    #selectable_row = range(box_info[2],box_info[2]+box_info[3])


    box_safe =0 #the number of boxes within the RetinaNet boundary
    box_corners = []
    counter = 0
    while ((box_safe <= num_boxes) and (counter < 1e5)):
        #loop until you get enough boxes to satisfy box_safe condition
        #or until you hit 1000, which means something is not right
        counter +=1 #safeguard for while

        #generate a random point within or some % outside of the box limits
        lowrow = row_corner- patch_gap
        highrow = row_limit + patch_gap
        assert(lowrow < highrow),print('Failure with random int ',lowrow,highrow)
        row_corner=np.random.randint(lowrow,highrow )
        
        lowcol = col_corner- patch_gap
        highcol = col_limit + patch_gap

        #debug if the random int problem shows
        assert(lowcol < highcol),print('Failure with random int ',lowcol,highcol)
        col_corner=np.random.randint(lowcol,highcol)
        
        #check to see if this new corner extends beyond any image end
        if ((row_corner < 0) or (col_corner <0) or 
            (((row_corner + patch_x) > irows) or ((col_corner + patch_y) > icols))):
            #box will extend off image boundary
            #don't let this new bounding box through, redo it
            #print('UNSAFE BOX--ADJUSTING ',row_corner,col_corner)
            #pass

            #check to see if our patch size doesn't go past boundary
            if  ((col_corner + patch_x)> icols):
                diff = (col_corner + patch_x) - icols
                col_corner = col_corner - diff
                #print('adjusting x limits of ', col_corner)
            
            if  ((row_corner + patch_y)> irows):
                diff = (row_corner + patch_y) - irows
                row_corner = row_corner - diff
                #print('adjusting y limits of ', col_corner)
            box_safe += 1
            box_corners.append([row_corner, col_corner])
        else: #This box is safe, use it for calculations
            box_safe += 1
            box_corners.append([row_corner, col_corner])

    if (box_safe < num_boxes):
        print('Failed to find enough sample boxes ', box_safe)
        if (counter > 1e5):
            print('Random Number selection not given enough samples. counter = ',counter)

    slice_lower = box_slice - 1
    slice_upper = box_slice + 1


    #store these random box corners off and return them to classify
    row_lims_low =  []
    row_lims_high = []
    col_lims_low =  []
    col_lims_high = []
    for ii in range(0,len(box_corners)):
        row_lims_low.append(box_corners[ii][0])
        row_lims_high.append(box_corners[ii][0] + patch_y)
        col_lims_low.append(box_corners[ii][1])
        col_lims_high.append(box_corners[ii][1] + patch_x)
        
        #row_lims_low, row_lims_high = box_corners[ii][0],box_corners[ii][0]+patch_y
        #col_lims_low, col_lims_high = box_corners[ii][1],box_corners[ii][1]+patch_x

    return row_lims_low, row_lims_high, col_lims_low, col_lims_high