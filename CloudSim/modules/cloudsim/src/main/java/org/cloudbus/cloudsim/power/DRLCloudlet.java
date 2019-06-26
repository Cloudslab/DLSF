package org.cloudbus.cloudsim.power;

import org.cloudbus.cloudsim.Cloudlet;
import org.cloudbus.cloudsim.UtilizationModel;

public class DRLCloudlet extends Cloudlet {

    /** Creation Delay **/
    public int delay = 0;

    public DRLCloudlet(
            final int cloudletId,
            final long cloudletLength,
            final int pesNumber,
            final long cloudletFileSize,
            final long cloudletOutputSize,
            final UtilizationModel utilizationModelCpu,
            final UtilizationModel utilizationModelRam,
            final UtilizationModel utilizationModelBw,
            final UtilizationModel utilizationModelDiskBw,
            final boolean record,
            final int delay) {
        super(cloudletId, cloudletLength, pesNumber, cloudletFileSize, cloudletOutputSize, utilizationModelCpu, utilizationModelRam,
                utilizationModelBw, utilizationModelDiskBw, record);
        this.delay = delay;
    }
}
