package org.cloudbus.cloudsim.examples.power.DeepRL;

import org.cloudbus.cloudsim.Cloudlet;
import org.cloudbus.cloudsim.UtilizationModel;
import org.cloudbus.cloudsim.UtilizationModelNull;
import org.cloudbus.cloudsim.UtilizationModelStochastic;
import org.cloudbus.cloudsim.examples.power.Constants;
import org.cloudbus.cloudsim.examples.power.random.RandomConstants;

import java.util.ArrayList;
import java.util.List;

/**
 * @author Shreshth Tuli
 * @since June 16, 2019
 */

public class DeepRLHelper {

    /**
     * Creates the cloudlet list.
     *
     * @param brokerId the broker id
     * @param cloudletsNumber the cloudlets number
     *
     * @return the list< cloudlet>
     */
    public static List<Cloudlet> createCloudletList(int brokerId, int cloudletsNumber) {
        List<Cloudlet> list = new ArrayList<Cloudlet>();

        long fileSize = 300;
        long outputSize = 300;
        long seed = DeepRLConstants.CLOUDLET_UTILIZATION_SEED;
        UtilizationModel utilizationModelNull = new UtilizationModelNull();

        for (int i = 0; i < cloudletsNumber; i++) {
            Cloudlet cloudlet;
            if (seed == -1) {
                cloudlet = new Cloudlet(
                        i,
                        Constants.CLOUDLET_LENGTH,
                        Constants.CLOUDLET_PES,
                        fileSize,
                        outputSize,
                        new UtilizationModelStochastic(),
                        utilizationModelNull,
                        utilizationModelNull);
            } else {
                cloudlet = new Cloudlet(
                        i,
                        Constants.CLOUDLET_LENGTH,
                        Constants.CLOUDLET_PES,
                        fileSize,
                        outputSize,
                        new UtilizationModelStochastic(seed * i),
                        utilizationModelNull,
                        utilizationModelNull);
            }
            cloudlet.setUserId(brokerId);
            cloudlet.setVmId(i);
            list.add(cloudlet);
        }

        return list;
    }
}
