package org.cloudbus.cloudsim.power;

import org.cloudbus.cloudsim.DatacenterCharacteristics;
import org.cloudbus.cloudsim.Storage;
import org.cloudbus.cloudsim.VmAllocationPolicy;

import java.util.List;

/**
 * The DRLDatacenter class implements functions for interaction
 * with the Deep Learning model and extends the PowerDatacenter class
 *
 * @author Shreshth Tuli
 * @since CloudSim Toolkit 5.0
 */

public class DRLDatacenter extends PowerDatacenter {

    /**
     * Instantiates a new DRLDatacenter.
     *
     * @param name               the datacenter name
     * @param characteristics    the datacenter characteristics
     * @param schedulingInterval the scheduling interval
     * @param vmAllocationPolicy the vm provisioner
     * @param storageList        the storage list
     * @throws Exception the exception
     */
    public DRLDatacenter(
            String name,
            DatacenterCharacteristics characteristics,
            VmAllocationPolicy vmAllocationPolicy,
            List<Storage> storageList,
            double schedulingInterval) throws Exception {
        super(name, characteristics, vmAllocationPolicy, storageList, schedulingInterval);
    }
}
