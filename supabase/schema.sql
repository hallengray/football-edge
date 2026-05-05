-- Football Edge schema
-- Run this in your Supabase SQL editor (Database → SQL Editor → New query)

create table if not exists predictions (
    id              bigserial primary key,
    fixture_id      text not null,
    home_team       text not null,
    away_team       text not null,
    kickoff         timestamptz not null,
    market          text not null,           -- 'h2h' | 'totals'
    outcome         text not null,           -- e.g. 'Home', 'Draw', 'Away', 'Over 2.5'
    model_probability    numeric not null,
    bookmaker       text not null,
    decimal_odds    numeric not null,
    value_pct       numeric not null,
    kelly_stake_fraction numeric not null,
    logged_at       timestamptz not null default now(),

    -- Upsert key: don't double-log the same prediction
    constraint predictions_unique_outcome unique (fixture_id, market, outcome)
);

create table if not exists bets (
    id              bigserial primary key,
    prediction_id   bigint not null references predictions(id) on delete cascade,
    stake_amount    numeric not null,
    notes           text default '',
    placed_at       timestamptz not null default now()
);

create table if not exists outcomes (
    id              bigserial primary key,
    bet_id          bigint not null references bets(id) on delete cascade,
    won             boolean not null,
    payout          numeric not null,        -- total returned (stake + winnings if won, 0 if lost)
    settled_at      timestamptz not null default now()
);

create index if not exists idx_predictions_kickoff on predictions(kickoff desc);
create index if not exists idx_predictions_fixture on predictions(fixture_id);
create index if not exists idx_bets_prediction on bets(prediction_id);
create index if not exists idx_outcomes_bet on outcomes(bet_id);
