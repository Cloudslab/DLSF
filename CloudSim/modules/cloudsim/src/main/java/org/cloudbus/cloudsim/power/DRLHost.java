package org.cloudbus.cloudsim.power;

import org.cloudbus.cloudsim.Pe;
import org.cloudbus.cloudsim.VmScheduler;
import org.cloudbus.cloudsim.power.models.PowerModel;
import org.cloudbus.cloudsim.provisioners.BwProvisioner;
import org.cloudbus.cloudsim.provisioners.RamProvisioner;

import java.util.List;

public class DRLHost extends PowerHost {

    /**
     * Instantiates a new PowerHost.
     *
     * @param id the id of the host
     * @param ramProvisioner the ram provisioner
     * @param bwProvisioner the bw provisioner
     * @param storage the storage capacity
     * @param peList the host's PEs list
     * @param vmScheduler the VM scheduler
     * @param powerModel the power model of host
     */
    public DRLHost(
            int id,
            RamProvisioner ramProvisioner,
            BwProvisioner bwProvisioner,
            long storage,
            List<? extends Pe> peList,
            VmScheduler vmScheduler,
            PowerModel powerModel) {
        super(id, ramProvisioner, bwProvisioner, storage, peList, vmScheduler, powerModel);
    }
}
