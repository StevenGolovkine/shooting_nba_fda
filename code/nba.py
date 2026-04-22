#!/usr/bin/env python
# -*-coding:utf8 -*
"""
Functions to scrap nba.com
--------------------------

"""

# Load packages
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from FDApy.representation.functional_data import DenseFunctionalData, BasisFunctionalData
from FDApy.preprocessing import MFPCA, UFPCA

class MidpointNormalize(mpl.colors.Normalize):
    def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False):
        self.vcenter = vcenter
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        # Note also that we must extrapolate beyond vmin/vmax
        x, y = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1.]
        return np.ma.masked_array(
            np.interp(value, x, y, left=-np.inf, right=np.inf)
        )

    def inverse(self, value):
        y, x = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1]
        return np.interp(value, x, y, left=-np.inf, right=np.inf)

class NbaScraper:
    """ Class to scrape data from the NBA official website.
    """
    @staticmethod
    def get_json_from_name(name: str, is_player=True) -> int:
        """ Get the json of a player or team from his name
        """
        from nba_api.stats.static import players, teams
        if is_player:
            nba_players = players.get_players()
            return [
                player for player in nba_players
                if player['full_name'] == name
            ][0]
        else:
            nba_teams = teams.get_teams()
            return [
                team for team in nba_teams if team['full_name'] == name
            ][0]
    
    @staticmethod
    def get_player_career(player_id: int) -> list:
        """ Get the career of a player from his id
        """
        from nba_api.stats.endpoints import playercareerstats
        career = playercareerstats.PlayerCareerStats(player_id=player_id)
        return career.get_data_frames()[0]
    
    @staticmethod
    def get_shot_data(id: int, team_ids: list, seasons: list) -> list:
        """ Get the shot data of a player from his id and seasons
        """
        from nba_api.stats.endpoints import shotchartdetail
        df = pd.DataFrame()
        for season in seasons:
            for team in team_ids:
                shot_data = shotchartdetail.ShotChartDetail(
                    team_id=team,
                    player_id=id,
                    context_measure_simple='FGA',
                    season_nullable=season
                )
                df = pd.concat([df, shot_data.get_data_frames()[0]])
        
        return df
    
    @staticmethod
    def get_all_ids(only_active=True) -> list:
        """ Get all the ids of the players
        """
        from nba_api.stats.static import players
        nba_players = players.get_players()
        if only_active:
            return [
                player['id'] for player in nba_players if player['is_active']
            ]
        return [player['id'] for player in nba_players]
    
    @staticmethod
    def get_player_headshot(id: int) -> str:
            """ Get the headshot of a player from his id
            """
            from nba_api.stats.static import players
            import requests
            import shutil
            
            url = f'https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{id}.png'
            output_path = f'../data/nba/transient/headshots/{id}.png'
            r = requests.get(url, stream=True)
            if r.status_code == 200:
                with open(output_path, 'wb') as f:
                    r.raw.decode_content = True
                    shutil.copyfileobj(r.raw, f)
    
    @staticmethod                                    
    def get_all_nba_headshots(only_active=False) -> None:
        """ Get the headshots of all the players
        """
        ids = NbaScraper.get_all_ids(only_active=only_active)
        for id in ids:
            NbaScraper.get_player_headshot(id)


class ShotCharts:
    def __init__(self) -> None:
        pass

    def create_court(ax: mpl.axes, color="white") -> mpl.axes:
        """ Create a basketball court in a matplotlib axes
        """
        # Short corner 3PT lines
        ax.plot([-220, -220], [0, 140], linewidth=2, color=color)
        ax.plot([220, 220], [0, 140], linewidth=2, color=color)
        # 3PT Arc
        ax.add_artist(
            mpl.patches.Arc(
                (0, 140), 440, 315,
                theta1=0, theta2=180,
                facecolor='none', edgecolor=color,
                lw=2
            )
        )
        # Lane and Key
        ax.plot([-80, -80], [0, 190], linewidth=2, color=color)
        ax.plot([80, 80], [0, 190], linewidth=2, color=color)
        ax.plot([-60, -60], [0, 190], linewidth=2, color=color)
        ax.plot([60, 60], [0, 190], linewidth=2, color=color)
        ax.plot([-80, 80], [190, 190], linewidth=2, color=color)
        ax.plot([-250, 250], [0, 0], linewidth=4, color='white')
        ax.add_artist(
            mpl.patches.Circle(
                (0, 190), 60, facecolor='none', edgecolor=color, lw=2
            )
        )
        # Rim
        ax.add_artist(
            mpl.patches.Circle(
                (0, 60), 15, facecolor='none', edgecolor=color, lw=2
            )
        )
        # Backboard
        ax.plot([-30, 30], [40, 40], linewidth=2, color=color)
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
        # Set axis limits
        ax.set_xlim(-250, 250)
        ax.set_ylim(0, 470)
        return ax

    def add_headshot(ax: mpl.axes, id: int) -> mpl.axes:
        from PIL import Image 
        from urllib import request
        BASE = "https://cdn.nba.com/headshots/nba/latest/260x190"
        headshot_path = f"{BASE}/{str(id)}.png"
        im = Image.open(request.urlopen(headshot_path))

        ax = ax.inset_axes([0.06, -0.04, 0.3, 0.3])
        ax.imshow(im)
        ax.axis('off')
        return ax

    def shots_chart_points(
        ax: mpl.axes,
        df: pd.DataFrame,
        name: str,
        position: str,
        add_headshot: bool = True
    ) -> mpl.axes:
        cond = df.PLAYER_NAME == name
        X = df[cond]['LOC_X'].to_numpy()
        Y = df[cond]['LOC_Y'].to_numpy()
        MADE = df[cond]['SHOT_MADE_FLAG'].to_numpy().astype('bool')
        idx = df[cond]['PLAYER_ID'].iloc[0]

        ax = ShotCharts.create_court(ax, 'black')
        ax.scatter(X[~MADE], Y[~MADE], c='#D81B60', marker='x', alpha=0.8, s=5, linewidth=0.2, label='Missed')
        ax.scatter(
            X[MADE], Y[MADE],
            edgecolors='#1E88E5', alpha=0.8, s=5,
            linewidth=0.2, facecolors='none', label='Made'
        )
        
        ax.text(0.03, 0.925, f"{name}", fontsize='large', transform=ax.transAxes)
        ax.text(0.03, 0.875, f"{position}", fontsize='medium', transform=ax.transAxes)
        
        leg = ax.legend(loc="upper right", title="")
        ax.add_artist(leg)
        
        if add_headshot:
            ax = ShotCharts.add_headshot(ax, idx)
        return ax

    def shots_chart(
        ax: mpl.axes,
        df: pd.DataFrame,
        name: str,
        title: str,
        add_headshot: bool = True
    ) -> mpl.axes:
        cond = df.PLAYER_NAME == name
        density = df[cond]['DENSITY'].iloc[0]
        idx = df[cond]['PLAYER_ID'].iloc[0]

        X_MIN, X_MAX = (-250, 250) 
        Y_MIN, Y_MAX = (0, 470)
        XX, YY = np.mgrid[X_MIN:X_MAX:201j, Y_MIN:Y_MAX:201j]

        midnorm = MidpointNormalize(vmin=-1., vcenter=0, vmax=1)
        ax.contourf(
            XX, YY, density / np.max(np.abs(density)),
            levels=30,
            cmap='seismic', norm=midnorm
        )
        ax.text(0.03, 0.925, f"{title}", fontsize='large', transform=ax.transAxes)
        ax = ShotCharts.create_court(ax, 'black')
        if add_headshot:
            ax = ShotCharts.add_headshot(ax, idx)
        return ax

    def shots_chart_2(
        ax_shots: mpl.axes,
        ax_shots_made: mpl.axes,
        df_shots: pd.DataFrame,
        df_shots_made: pd.DataFrame,
        name: str,
        title_shots: str,
        title_shots_made: str,
        add_headshot: bool = True
    ) -> np.ndarray:
        cond = df_shots.PLAYER_NAME == name

        density_shots = df_shots[cond]['DENSITY'].iloc[0]
        density_shots_made = df_shots_made[cond]['DENSITY'].iloc[0]
        idx_shots = df_shots[cond]['PLAYER_ID'].iloc[0]
        idx_shots_made = df_shots_made[cond]['PLAYER_ID'].iloc[0]

        maximum = max(
            np.max(np.abs(density_shots)),
            np.max(np.abs(density_shots_made))
        )

        X_MIN, X_MAX = (-250, 250)
        Y_MIN, Y_MAX = (0, 470)
        XX, YY = np.mgrid[X_MIN:X_MAX:201j, Y_MIN:Y_MAX:201j]

        midnorm = MidpointNormalize(vmin=-1., vcenter=0, vmax=1)
        
        # Shots missed
        ax_shots.contourf(
            XX, YY, density_shots / maximum,
            levels=30,
            cmap='seismic', norm=midnorm
        )
        ax_shots.text(
            0.03, 0.925, f"{title_shots}",
            fontsize='large', transform=ax_shots.transAxes
        )
        ax_shots = ShotCharts.create_court(ax_shots, 'black')
        if add_headshot:
            ax_shots = ShotCharts.add_headshot(ax_shots, idx_shots)

        # Shots made
        ax_shots_made.contourf(
            XX, YY, density_shots_made / maximum,
            levels=30,
            cmap='seismic', norm=midnorm
        )
        ax_shots_made.text(
            0.03, 0.925, f"{title_shots_made}",
            fontsize='large', transform=ax_shots_made.transAxes
        )
        ax_shots_made = ShotCharts.create_court(ax_shots_made, 'black')
        if add_headshot:
            ax_shots_made = ShotCharts.add_headshot(ax_shots_made, idx_shots_made)
        return ax_shots, ax_shots_made

    def shots_chart_fdata(
        ax: mpl.axes,
        fdata: DenseFunctionalData,
        title: str,
        maximum: float
    ) -> mpl.axes:
        density = fdata.values[0]
    
        X_MIN, X_MAX = (-250, 250) 
        Y_MIN, Y_MAX = (0, 470)
        XX, YY = np.mgrid[X_MIN:X_MAX:201j, Y_MIN:Y_MAX:201j]

        midnorm = MidpointNormalize(vmin=-1., vcenter=0, vmax=1)
        ax.contourf(
            XX, YY, density / maximum,
            levels=30,
            cmap='seismic', norm=midnorm
        )
        ax.text(0.03, 0.925, f"{title}", fontsize='large', transform=ax.transAxes)
        ax = ShotCharts.create_court(ax, 'black')
        return ax

    def shots_chart_reconstruction(
        ax: mpl.axes,
        df: pd.DataFrame,
        df_reconstruction: DenseFunctionalData,
        name: str,
        title: str,
        add_headshot: bool = True
    ) -> mpl.axes:
        cond = df.PLAYER_NAME == name
        idx = df[cond]['PLAYER_ID'].iloc[0]
        indice = df.index[cond]
        density = df_reconstruction[indice[0]].values.squeeze()

        X_MIN, X_MAX = (-250, 250) 
        Y_MIN, Y_MAX = (0, 470)
        XX, YY = np.mgrid[X_MIN:X_MAX:201j, Y_MIN:Y_MAX:201j]

        midnorm = MidpointNormalize(vmin=-1., vcenter=0, vmax=1)
        ax.contourf(
            XX, YY, density / np.max(np.abs(density)),
            levels=30,
            cmap='seismic', norm=midnorm
        )
        ax.text(0.03, 0.925, f"{title}", fontsize='large', transform=ax.transAxes)
        ax = ShotCharts.create_court(ax, 'black')
        if add_headshot:
            ax = ShotCharts.add_headshot(ax, idx)
        return ax

    def shots_chart_reconstruction_2(
        ax_shots: mpl.axes,
        ax_shots_made: mpl.axes,
        df_shots: pd.DataFrame,
        df_shots_made: pd.DataFrame,
        df_reconstruction_shots: DenseFunctionalData,
        df_reconstruction_shots_made: DenseFunctionalData,
        name: str,
        title_shots: str,
        title_shots_made: str,
        add_headshot: bool = True
    ) -> tuple[mpl.axes, mpl.axes]:
        cond_shots = df_shots.PLAYER_NAME == name
        cond_shots_made = df_shots_made.PLAYER_NAME == name

        idx_shots = df_shots[cond_shots]['PLAYER_ID'].iloc[0]
        idx_shots_made = df_shots_made[cond_shots_made]['PLAYER_ID'].iloc[0]
        indice_shots = df_shots.index[cond_shots]
        indice_shots_made = df_shots_made.index[cond_shots_made]
        density_shots = df_reconstruction_shots[indice_shots[0]].values.squeeze()
        density_shots_made = (
            df_reconstruction_shots_made[indice_shots_made[0]].values.squeeze()
        )

        maximum = max(
            np.max(np.abs(density_shots)),
            np.max(np.abs(density_shots_made))
        )

        X_MIN, X_MAX = (-250, 250)
        Y_MIN, Y_MAX = (0, 470)
        XX, YY = np.mgrid[X_MIN:X_MAX:201j, Y_MIN:Y_MAX:201j]

        midnorm = MidpointNormalize(vmin=-1., vcenter=0, vmax=1)
        ax_shots.contourf(
            XX, YY, density_shots / maximum,
            levels=30,
            cmap='seismic', norm=midnorm
        )
        ax_shots.text(
            0.03, 0.925, f"{title_shots}",
            fontsize='large', transform=ax_shots.transAxes
        )
        ax_shots = ShotCharts.create_court(ax_shots, 'black')
        if add_headshot:
            ax_shots = ShotCharts.add_headshot(ax_shots, idx_shots)

        ax_shots_made.contourf(
            XX, YY, density_shots_made / maximum,
            levels=30,
            cmap='seismic', norm=midnorm
        )
        ax_shots_made.text(
            0.03, 0.925, f"{title_shots_made}",
            fontsize='large', transform=ax_shots_made.transAxes
        )
        ax_shots_made = ShotCharts.create_court(ax_shots_made, 'black')
        if add_headshot:
            ax_shots_made = ShotCharts.add_headshot(ax_shots_made, idx_shots_made)

        return ax_shots, ax_shots_made

    def mean_chart(
        ax: mpl.axes,
        mfpca: MFPCA,
        idx_c: int,
        title: str
    ) -> mpl.axes:
        if isinstance(mfpca, MFPCA):
            mean = mfpca.mean.data[idx_c][0].values.squeeze()
        else:
            mean = mfpca.mean[0].values.squeeze()

        X_MIN, X_MAX = (-250, 250) 
        Y_MIN, Y_MAX = (0, 470)
        XX, YY = np.mgrid[X_MIN:X_MAX:201j, Y_MIN:Y_MAX:201j]

        midnorm = MidpointNormalize(vmin=-1., vcenter=0, vmax=1)
        ax.contourf(
            XX, YY, mean / np.max(np.abs(mean)),
            levels=30,
            cmap='seismic', norm=midnorm
        )
        ax.text(
            0.03, 0.925, f"{title}", fontsize='large', transform=ax.transAxes
        )
        ax = ShotCharts.create_court(ax, 'black')
        return ax

    def mean_chart_2(
        ax_mean_1: mpl.axes,
        ax_mean_2: mpl.axes,
        mfpca: MFPCA,
        title_1: str,
        title_2: str
    ) -> tuple[mpl.axes, mpl.axes]:
        if isinstance(mfpca, MFPCA):
            mean_1 = mfpca.mean.data[0][0].values.squeeze()
            mean_2 = mfpca.mean.data[1][0].values.squeeze()
        else:
            mean_1 = mfpca.mean[0].values.squeeze()
            mean_2 = mfpca.mean[1].values.squeeze()

        maximum = max(
            np.max(np.abs(mean_1)),
            np.max(np.abs(mean_2))
        )

        X_MIN, X_MAX = (-250, 250)
        Y_MIN, Y_MAX = (0, 470)
        XX, YY = np.mgrid[X_MIN:X_MAX:201j, Y_MIN:Y_MAX:201j]

        midnorm = MidpointNormalize(vmin=-1., vcenter=0, vmax=1)

        ax_mean_1.contourf(
            XX, YY, mean_1 / maximum,
            levels=30,
            cmap='seismic', norm=midnorm
        )
        ax_mean_1.text(
            0.03, 0.925, f"{title_1}",
            fontsize='large', transform=ax_mean_1.transAxes
        )
        ax_mean_1 = ShotCharts.create_court(ax_mean_1, 'black')

        ax_mean_2.contourf(
            XX, YY, mean_2 / maximum,
            levels=30,
            cmap='seismic', norm=midnorm
        )
        ax_mean_2.text(
            0.03, 0.925, f"{title_2}",
            fontsize='large', transform=ax_mean_2.transAxes
        )
        ax_mean_2 = ShotCharts.create_court(ax_mean_2, 'black')

        return ax_mean_1, ax_mean_2

    def components_chart(
        ax: mpl.axes,
        mfpca: MFPCA,
        idx: int,
        idx_c: int,
        title: str
    ) -> mpl.axes:
        if isinstance(mfpca, MFPCA):
            if isinstance(mfpca.eigenfunctions.data[idx_c], BasisFunctionalData):
                eigenfunctions = mfpca.eigenfunctions.data[idx_c].to_grid()
            else:
                eigenfunctions = mfpca.eigenfunctions.data[idx_c]
        else:
            eigenfunctions = mfpca.eigenfunctions
        eigenfunctions = eigenfunctions[idx].values.squeeze()
        pct = 100 * mfpca.eigenvalues / np.sum(mfpca.eigenvalues)

        X_MIN, X_MAX = (-250, 250) 
        Y_MIN, Y_MAX = (0, 470)
        XX, YY = np.mgrid[X_MIN:X_MAX:201j, Y_MIN:Y_MAX:201j]

        midnorm = MidpointNormalize(vmin=-1., vcenter=0, vmax=1)
        
        ax.contourf(
            XX, YY, eigenfunctions / np.max(np.abs(eigenfunctions)),
            levels=30,
            cmap='seismic', norm=midnorm
        )
        ax.text(
            0.03, 0.925, f"{title}", fontsize='large', transform=ax.transAxes
        )
        ax = ShotCharts.create_court(ax, 'black')
        return ax

    def components_chart_2(
        ax_component_1: mpl.axes,
        ax_component_2: mpl.axes,
        mfpca: MFPCA,
        idx: int,
        title_1: str,
        title_2: str
    ) -> tuple[mpl.axes, mpl.axes]:
        if isinstance(mfpca, MFPCA):
            if isinstance(mfpca.eigenfunctions.data[0], BasisFunctionalData):
                eigenfunctions_1 = mfpca.eigenfunctions.data[0].to_grid()
            else:
                eigenfunctions_1 = mfpca.eigenfunctions.data[0]

            if isinstance(mfpca.eigenfunctions.data[1], BasisFunctionalData):
                eigenfunctions_2 = mfpca.eigenfunctions.data[1].to_grid()
            else:
                eigenfunctions_2 = mfpca.eigenfunctions.data[1]
        else:
            eigenfunctions_1 = mfpca.eigenfunctions[0]
            eigenfunctions_2 = mfpca.eigenfunctions[1]

        eigenfunctions_1 = eigenfunctions_1[idx].values.squeeze()
        eigenfunctions_2 = eigenfunctions_2[idx].values.squeeze()

        maximum = max(
            np.max(np.abs(eigenfunctions_1)),
            np.max(np.abs(eigenfunctions_2))
        )

        X_MIN, X_MAX = (-250, 250)
        Y_MIN, Y_MAX = (0, 470)
        XX, YY = np.mgrid[X_MIN:X_MAX:201j, Y_MIN:Y_MAX:201j]

        midnorm = MidpointNormalize(vmin=-1., vcenter=0, vmax=1)

        ax_component_1.contourf(
            XX, YY, eigenfunctions_1 / maximum,
            levels=30,
            cmap='seismic', norm=midnorm
        )
        ax_component_1.text(
            0.03, 0.925, f"{title_1}",
            fontsize='large', transform=ax_component_1.transAxes
        )
        ax_component_1 = ShotCharts.create_court(ax_component_1, 'black')

        ax_component_2.contourf(
            XX, YY, eigenfunctions_2 / maximum,
            levels=30,
            cmap='seismic', norm=midnorm
        )
        ax_component_2.text(
            0.03, 0.925, f"{title_2}",
            fontsize='large', transform=ax_component_2.transAxes
        )
        ax_component_2 = ShotCharts.create_court(ax_component_2, 'black')

        return ax_component_1, ax_component_2

    def shots_decomposition_chart(
        ax: mpl.axes,
        mfpca: MFPCA,
        scores: np.ndarray,
        idx: int,
        idx_c: int,
        name: str,
        title: str,
        maximum: float
    ) -> plt.figure:
        if isinstance(mfpca, MFPCA):
            if isinstance(mfpca.eigenfunctions.data[idx_c], BasisFunctionalData):
                eigenfunctions = mfpca.eigenfunctions.data[idx_c].to_grid()
            else:
                eigenfunctions = mfpca.eigenfunctions.data[idx_c]
        else:
            eigenfunctions = mfpca.eigenfunctions
        eigenfunctions = eigenfunctions[idx].values.squeeze()
        pct = 100 * mfpca.eigenvalues / np.sum(mfpca.eigenvalues)
        score = scores.loc[scores.PLAYER_NAME == name, idx].values
        fpc = score * eigenfunctions
        
        X_MIN, X_MAX = (-250, 250) 
        Y_MIN, Y_MAX = (0, 470)
        XX, YY = np.mgrid[X_MIN:X_MAX:201j, Y_MIN:Y_MAX:201j]
        
        midnorm = MidpointNormalize(vmin=-1., vcenter=0, vmax=1)
        ax.contourf(
            XX, YY, fpc / maximum,
            levels=30,
            cmap='seismic', norm=midnorm,
        )
        ax = ShotCharts.create_court(ax, 'black')
        ax.text(0.03, 0.925, f"{title}", fontsize='large', transform=ax.transAxes)
        return ax

    def shots_decomposition_cluster(
        ax: mpl.axes,
        mfpca: MFPCA,
        score: np.ndarray,
        idx: int,
        idx_c: int,
        title: str,
        maximum: float
    ) -> plt.figure:
        if isinstance(mfpca, MFPCA):
            if isinstance(mfpca.eigenfunctions.data[idx_c], BasisFunctionalData):
                eigenfunctions = mfpca.eigenfunctions.data[idx_c].to_grid()
            else:
                eigenfunctions = mfpca.eigenfunctions.data[idx_c]
        else:
            eigenfunctions = mfpca.eigenfunctions
        eigenfunctions = eigenfunctions[idx].values.squeeze()
        pct = 100 * mfpca.eigenvalues / np.sum(mfpca.eigenvalues)
        fpc = score * eigenfunctions
        
        X_MIN, X_MAX = (-250, 250) 
        Y_MIN, Y_MAX = (0, 470)
        XX, YY = np.mgrid[X_MIN:X_MAX:201j, Y_MIN:Y_MAX:201j]
        
        midnorm = MidpointNormalize(vmin=-1., vcenter=0, vmax=1)
        ax.contourf(
            XX, YY, fpc / maximum,
            levels=30,
            cmap='seismic', norm=midnorm,
        )
        ax = ShotCharts.create_court(ax, 'black')
        ax.text(0.03, 0.925, f"{title}", fontsize='large', transform=ax.transAxes)
        return ax
