import json
import datetime

# Load the data
with open('nba_contests.json', 'r') as f:
    data = json.load(f)

# Convert timestamp to readable date
def convert_timestamp(ts):
    # Extract timestamp from /Date(XXXXXXXXXXXX)/ format
    timestamp = int(ts.split('(')[1].split(')')[0])
    dt = datetime.datetime.fromtimestamp(timestamp/1000)
    return dt.strftime('%Y-%m-%d %H:%M:%S')

def is_gpp_tournament(contest):
    """
    Determine if a contest is a GPP (Guaranteed Prize Pool) tournament
    based on various attributes
    """
    # Must be Classic game type
    if contest.get('gameType') != 'Classic':
        return False

    # Check contest name for indicators of non-GPP formats FIRST
    name_lower = contest['n'].lower()
    exclude_terms = ['double up', '50/50', 'fifty fifty', 'head to head', 'h2h',
                    'satellite', 'qualifier', 'ticket', 'winner take all', 'wta',
                    'winner takes all', 'heavy hitter', '3-player', '4-player',
                    '2-player', 'must fill']

    for term in exclude_terms:
        if term in name_lower:
            return False

    # Additional filters to exclude non-GPP formats
    if 'attr' in contest:
        # Exclude Double Ups, 50/50s, and other cash formats
        if contest['attr'].get('IsDoubleUp') == 'true':
            return False
        if contest['attr'].get('IsFiftyfifty') == 'true':
            return False

    # Must be a guaranteed tournament to be considered a GPP
    if 'attr' in contest and 'IsGuaranteed' in contest['attr']:
        if contest['attr']['IsGuaranteed'] == 'true':
            return True

    return False

print('Total NBA contests found: {}'.format(len(data['Contests'])))
print()

# Filter for Classic GPP tournaments on 2025-10-21
target_date = '2025-10-21'
classic_gpps = []

for contest in data['Contests']:
    contest_date = convert_timestamp(contest['sd'])
    if contest_date.startswith(target_date) and is_gpp_tournament(contest):
        classic_gpps.append(contest)

print('Classic GPP tournaments on {}: {}'.format(target_date, len(classic_gpps)))
print()

# Sort by prize pool (descending)
classic_gpps.sort(key=lambda x: x['po'], reverse=True)

# Show details for Classic GPP tournaments
for i, contest in enumerate(classic_gpps[:20]):  # Show top 20
    print('{}. {}'.format(i+1, contest['n']))
    print('   Start: {}'.format(convert_timestamp(contest['sd'])))
    print('   Prize Pool: ${:,.0f}'.format(contest['po']))
    print('   Entry Fee: ${}'.format(contest['a']))
    print('   Max Entries: {}'.format(contest.get('mec', 'Unlimited')))
    print('   Current Entries: {}'.format(contest.get('m', 0)))
    print('   Guaranteed: {}'.format(contest['attr'].get('IsGuaranteed', 'false') if 'attr' in contest else 'false'))
    print()

print('Summary statistics:')
total_prize_pool = sum(contest['po'] for contest in classic_gpps)
avg_entry_fee = sum(contest['a'] for contest in classic_gpps) / len(classic_gpps) if classic_gpps else 0
print('Total prize pool across all Classic GPPs: ${:,.0f}'.format(total_prize_pool))
print('Average entry fee: ${:.2f}'.format(avg_entry_fee))
print('Number of Classic GPP tournaments: {}'.format(len(classic_gpps)))

# Analyze entry fee distribution
fee_ranges = {'$0-5': 0, '$5-10': 0, '$10-25': 0, '$25-50': 0, '$50-100': 0, '$100+': 0}
for contest in classic_gpps:
    fee = contest['a']
    if fee <= 5:
        fee_ranges['$0-5'] += 1
    elif fee <= 10:
        fee_ranges['$5-10'] += 1
    elif fee <= 25:
        fee_ranges['$10-25'] += 1
    elif fee <= 50:
        fee_ranges['$25-50'] += 1
    elif fee <= 100:
        fee_ranges['$50-100'] += 1
    else:
        fee_ranges['$100+'] += 1

print('\nEntry fee distribution:')
for fee_range, count in fee_ranges.items():
    print('  {}: {} tournaments'.format(fee_range, count))