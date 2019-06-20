package org.cloudbus.cloudsim.cost;

import org.cloudbus.cloudsim.Host;
import org.cloudbus.cloudsim.Pe;
import org.cloudbus.cloudsim.VmScheduler;
import org.cloudbus.cloudsim.cost.models.CostModel;
import org.cloudbus.cloudsim.provisioners.BwProvisioner;
import org.cloudbus.cloudsim.provisioners.RamProvisioner;

import java.util.List;

public class CostHost extends Host {

    /** Cost Model of this host **/
    private CostModel costModel;

    /**
     * Instantiates a new host.
     *
     * @param id the host id
     * @param ramProvisioner the ram provisioner
     * @param bwProvisioner the bw provisioner
     * @param storage the storage capacity
     * @param peList the host's PEs list
     * @param vmScheduler the vm scheduler
     */
    public CostHost(
            int id,
            RamProvisioner ramProvisioner,
            BwProvisioner bwProvisioner,
            long storage,
            List<? extends Pe> peList,
            VmScheduler vmScheduler,
            CostModel costModel) {
        super(id, ramProvisioner, bwProvisioner, storage, peList, vmScheduler);
        this.setCostModel(costModel);
    }


    /**
     * Get cost model of this host
     *
     * @return cost model
     */
    protected CostModel getCostModel(){
        return costModel;
    }

    /**
     * Set cost model of this host
     *
     * @param cm cost model to set
     */
    protected void setCostModel(CostModel cm){
        this.costModel = cm;
    }
}
