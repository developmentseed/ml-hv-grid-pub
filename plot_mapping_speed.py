"""
plot_mapping_speed.py

@author: developmentseed

Simple plots to show time spent mapping before and after ML predictions are
available.
"""
import os.path as op
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns

from config import plot_dir

plt.close('all')
sns.set(style="whitegrid")
#sns.set_style('darkgrid', {"axes.facecolor": ".9"})
sns.set_context('talk', font_scale=1.1)

# Load data
mapping_rates = pd.read_csv(op.join(plot_dir, 'mapping_rates.csv'))
palette = ['#4477AA', '#117733', '#DDCC77', '#CC6677']
avg_indices = [6, 7]
properties = dict(x='Mapping scheme', palette=palette, data=mapping_rates,
                  size=3, kind='bar')

##################
# Plot mapping rate
##################
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7.5, 10), sharex=True)

sns.factorplot(y="km^2 per hour", hue='Country', ax=ax1, **properties)
ax1.set_ylabel(r'km$^2$ per hour')

# Substations per hour
sns.factorplot(y="Substations per hour", hue='Country', ax=ax2, **properties)

# Towers per hour
sns.factorplot(y="Towers per hour", hue='Country', ax=ax3, **properties)


# Set hatch on averages
for ax in [ax1, ax2, ax3]:
    for bi, bar in enumerate(ax.patches):
        if bi in avg_indices:
            bar.set_hatch('//')

ax1.legend(frameon=True)

# Remove x-labels on top plots
ax1.set_xlabel('')
ax2.set_xlabel('')

# Remove legend on all but top plot
ax2.legend().set_visible(False)
ax3.legend().set_visible(False)

sns.despine(left=True, top=True)
plt.tight_layout(h_pad=5)
ax1.set_title('Manual and ML-assisted mapping rates')
fig.savefig(op.join(plot_dir, 'mapping_rate.png'), dpi=150)

##############################################################################
# Number of features mapped
##############################################################################

# Load data
mapping_features = pd.read_csv(op.join(plot_dir, 'mapping_features_added.csv'))
properties = dict(x='Feature', palette=palette, data=mapping_features,
                  size=3, kind='bar')

########################################
# Total features at project completion
########################################
fig2, (ax2_1, ax2_2, ax2_3) = plt.subplots(1, 3, figsize=(11, 5))
sns.set_color_codes("pastel")
sns.barplot(x='Country', y='Total km HV line', data=mapping_features,
            ax=ax2_1, color='b', label="Total at project's end")
sns.barplot(x='Country', y='Total substations', data=mapping_features,
            ax=ax2_2, color='b')
sns.barplot(x='Country', y='Total towers', data=mapping_features, ax=ax2_3,
            color='b')

########################################
# Total mapped by the data team
########################################
sns.set_color_codes("muted")
sns.barplot(x='Country', y='Existing km HV line', data=mapping_features,
            ax=ax2_1, color='b', label='Previously existing')
sns.barplot(x='Country', y='Existing substations', data=mapping_features,
            ax=ax2_2, color='b')
sns.barplot(x='Country', y='Existing towers', data=mapping_features,
            ax=ax2_3, color='b')
ax2_3.set_ylim(0, 1e5)

# Set labels
ax2_1.set_ylabel('Km of HV line')
ax2_2.set_ylabel('Number of substations')
ax2_3.set_ylabel('Number of towers')
for ax in [ax2_1, ax2_2, ax2_3]:
    ax.tick_params(axis='x', rotation=60, pad=0)

    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_horizontalalignment('right')



ax2_1.legend(loc="upper right", bbox_to_anchor=(1.555, 1.05), frameon=True,
             fontsize=13, shadow=True)
plt.tight_layout(w_pad=3)
sns.despine(left=True, top=True)

fig2.savefig(op.join(plot_dir, 'mapped_features.png'), dpi=150)
