import h5py
import os

# class to convert datasets into hdf5 format.


class HDF5DatasetWriter:
    def __init__(self, dims, outputPath, dataKey="images", bufSize=1000):
        """

        :param dims: shape of the data we stored. same as numpy.shape. i.e. store CIFAR-10 (60000, 32, 32, 3)
        :param outputPath: path to save .hdf5
        :param dataKey: name of dataset file. dataKey.hdf5
        :param bufSize: number of samples before flushing data in memory to
        """
        # confirm output path does not exists
        if os.path.exists(outputPath):
            raise ValueError("The supplied 'output' path laready exists and cannot be overwritten."
                             " Manually delete the file before continuing.", outputPath)

        # Open the HDF5 database for writing and create two datasets:
        # 1. to store the image/features
        # 2. store the class labels
        self. db = h5py.File(outputPath, "w")
        self.data = self.db.create_dataset(dataKey, dims, dtype="float")
        self.labels = self.db.create_dataset("labels", (dims[0],), dtype="int")

        # store the buffer size, then init the buffer itself along with the index into the database
        self.bufSize = bufSize
        self.buffer = {"data": [], "labels": []}
        self.idx = 0

    def add(self, rows, labels):
        """
        Add data to buffer
        :param rows: row to add to data set
        :param labels: label of row
        :return: None
        """

        self.buffer["data"].extend(rows)
        self.buffer["labels"].extend(labels)

        # check if data buffer full
        if len(self.buffer["data"]) >= self.bufSize:
            self.flush()

    def flush(self):
        """
        write the buffer to disk then reset buffer
        :return: None
        """
        i = self.idx + len(self.buffer["data"])
        self.data[self.idx:i] = self.buffer["data"]
        self.labels[self.idx:i] = self.buffer["labels"]
        self.idx = i
        self.buffer = {"data": [], "labels":[]}

    def storeClassLabels(self, classLabels):
        """
        create a dataset to store the actual class label names then store the class labels
        :param classLabels:
        :return:
        """
        dt = h5py.special_dtype(vlen=str)
        labelSet = self.db.create_dataset("label_names",
                                          (len(classLabels),),
                                          dtype=dt)
        labelSet[:] = classLabels

    def close(self):
        """
        write remaining dataset in buffer to hdf5 and close files
        :return:
        """
        if len(self.buffer["data"]) > 0:
            self.flush()

        self.db.close()



