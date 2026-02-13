"""
FPL Review Client - Headless Patreon login and CSV download.

Uses Playwright to authenticate via Patreon OAuth and download
projected points CSV from fplreview.com.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional

from playwright.async_api import async_playwright, Page, Browser


class FPLReviewClient:
    """Client for downloading data from fplreview.com via Patreon auth."""

    FPLREVIEW_URL = "https://fplreview.com"
    MASSIVE_DATA_PLANNER_URL = "https://fplreview.com/massive-data-planner/"
    FREE_PLANNER_URL = "https://fplreview.com/free-planner/"

    def __init__(
        self,
        email: str,
        password: str,
        download_dir: Path,
        logger: Optional[logging.Logger] = None,
        team_id: str = "",
    ):
        self.email = email
        self.password = password
        self.download_dir = download_dir
        self.logger = logger or logging.getLogger(__name__)
        self.team_id = str(team_id) if team_id else ""

    def _is_fplreview_url(self, url: str) -> bool:
        """Check if URL's domain is fplreview.com (not just mentioned in query params)."""
        from urllib.parse import urlparse
        return urlparse(url).netloc.lower() in ("fplreview.com", "www.fplreview.com")

    def _is_patreon_url(self, url: str) -> bool:
        """Check if URL's domain is patreon.com."""
        from urllib.parse import urlparse
        return "patreon.com" in urlparse(url).netloc.lower()

    async def download_projections_csv(self, headless: bool = True) -> Optional[Path]:
        """
        Login to FPL Review via Patreon and download the projections CSV.

        Args:
            headless: Run browser in headless mode (default True)

        Returns:
            Path to downloaded CSV file, or None if download failed.
        """
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            self.logger.error(
                "Playwright not installed. Run: pip install playwright && playwright install chromium"
            )
            return None

        self.download_dir.mkdir(parents=True, exist_ok=True)
        self._headless = headless

        # Persist browser profile to maintain Patreon login session across runs
        # (avoids email code verification on every run from VPS)
        # Delete data/browser_profile/ manually if session becomes corrupted
        user_data_dir = self.download_dir / "browser_profile"
        user_data_dir.mkdir(parents=True, exist_ok=True)

        async with async_playwright() as p:
            # Use persistent context to maintain login state
            # Set downloads path to our data directory
            context = await p.chromium.launch_persistent_context(
                user_data_dir=str(user_data_dir),
                headless=headless,
                accept_downloads=True,
                downloads_path=str(self.download_dir),
                viewport={"width": 1280, "height": 800},
                # Make browser look more human-like
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            )
            page = context.pages[0] if context.pages else await context.new_page()

            try:
                # Step 1: Navigate to FPL Review
                self.logger.info("Navigating to FPL Review...")
                await page.goto(self.FPLREVIEW_URL, wait_until="domcontentloaded", timeout=30000)

                # Step 2: Try Patreon login (not required for Free Planner)
                logged_in = await self._login_via_patreon(page)
                if not logged_in:
                    self.logger.warning("Patreon login failed - continuing with Free Planner (no login required)")

                # Step 3: Navigate to FPL Review Free Planner
                self.logger.info("Navigating to FPL Review...")
                await page.goto(self.FPLREVIEW_URL, wait_until="domcontentloaded", timeout=60000)
                await page.wait_for_timeout(2000)

                # Handle cookie consent popup if present
                await self._accept_cookies(page)

                # Step 4: Navigate to the planner via menu
                await self._navigate_to_planner(page)

                # Step 5: Check if we need to reconnect Patreon (only if logged in)
                if logged_in:
                    reconnected = await self._handle_fplreview_reconnect(page)
                    if reconnected:
                        # Navigate back to planner after reconnect
                        await self._navigate_to_planner(page)

                # Step 5.5: Verify data is loaded before attempting download
                data_ready = await self._wait_for_table_data(page, timeout_s=10)
                if not data_ready:
                    self.logger.warning("Data not loaded yet, retrying data trigger...")
                    await self._trigger_data_load(page)

                # Step 6: Download CSV
                csv_path = await self._download_csv(page, context)
                return csv_path

            except Exception as e:
                self.logger.error(f"Error during download: {e}")
                # Save screenshot for debugging
                screenshot_path = self.download_dir / "fplreview_error.png"
                await page.screenshot(path=str(screenshot_path))
                self.logger.info(f"Screenshot saved to {screenshot_path}")
                return None

            finally:
                await context.close()

    async def _login_via_patreon(self, page: Page) -> bool:
        """Handle Patreon OAuth login flow."""
        try:
            # First check if we're already logged in (User Settings visible instead of Patreon Login)
            try:
                user_settings = page.locator('a:has-text("User Settings"), text="User Settings"').first
                if await user_settings.is_visible(timeout=3000):
                    self.logger.info("Already logged in (User Settings visible in nav)")
                    return True
            except:
                pass

            # Check if "Patreon Login" text is visible in nav (means NOT logged in)
            try:
                patreon_login_nav = page.locator('a:has-text("Patreon Login")').first
                if not await patreon_login_nav.is_visible(timeout=2000):
                    self.logger.info("Already logged in (no Patreon Login in nav)")
                    return True
            except:
                # If we can't find Patreon Login, we might be logged in
                pass

            # Look for Patreon LOGIN button specifically (not just any Patreon link)
            self.logger.info("Looking for Patreon login button...")

            # Be more specific - look for login-related Patreon buttons only
            patreon_selectors = [
                'a:has-text("Patreon Login")',
                'a:has-text("Login with Patreon")',
                'a:has-text("Log in with Patreon")',
                'button:has-text("Patreon Login")',
                '.patreon-login',
            ]

            patreon_btn = None
            for selector in patreon_selectors:
                try:
                    patreon_btn = page.locator(selector).first
                    if await patreon_btn.is_visible(timeout=2000):
                        self.logger.info(f"Found Patreon login button with selector: {selector}")
                        break
                except:
                    continue

            if not patreon_btn:
                # No login button found - might already be logged in
                self.logger.info("No Patreon login button found - assuming already logged in")
                return True

            # Click Patreon login button
            self.logger.info("Clicking Patreon login button...")
            await patreon_btn.click()

            # Wait for redirect
            await page.wait_for_timeout(3000)

            # Check if we're on Patreon's domain
            current_url = page.url
            self.logger.info(f"Current URL after click: {current_url}")

            if self._is_patreon_url(current_url):
                # We're on Patreon login page
                return await self._complete_patreon_login(page)
            elif self._is_fplreview_url(current_url):
                # We were already logged in and got redirected back
                self.logger.info("Already authenticated via Patreon (redirected back to FPL Review)")
                return True
            else:
                # Maybe there's a popup or redirect
                self.logger.info(f"Unexpected URL: {current_url[:100]}")
                return False

        except Exception as e:
            self.logger.error(f"Error during Patreon login: {e}")
            return False

    async def _handle_cloudflare(self, page: Page) -> bool:
        """Handle Cloudflare challenge if present. Returns True if handled successfully."""
        try:
            # Check if we're on a Cloudflare challenge page
            if "challenge" in page.url or await page.locator("text=Verify you are human").is_visible(timeout=2000):
                self.logger.warning("Cloudflare CAPTCHA detected!")
                self.logger.info("Waiting for CAPTCHA to be solved (up to 60 seconds)...")
                self.logger.info("If running headless, try: python -m src.fplreview_client (runs non-headless)")

                # Wait for the challenge to be completed (page will redirect)
                for _ in range(60):
                    await page.wait_for_timeout(1000)
                    if "challenge" not in page.url and not await page.locator("text=Verify you are human").is_visible(timeout=500):
                        self.logger.info("Cloudflare challenge passed!")
                        return True

                self.logger.error("Cloudflare CAPTCHA timeout - please run in non-headless mode first")
                await page.screenshot(path=str(self.download_dir / "cloudflare_timeout.png"))
                return False
            return True
        except:
            return True  # No Cloudflare challenge detected

    async def _complete_patreon_login(self, page: Page) -> bool:
        """Complete the Patreon login form."""
        try:
            # Handle Cloudflare if present
            if not await self._handle_cloudflare(page):
                return False

            # Dismiss any cookie consent popups (Transcend manager)
            await self._accept_cookies(page)

            self.logger.info("Completing Patreon login form...")

            # Step 1: Wait for and fill email input
            email_input = page.locator('input[type="email"], input[name="email"], input[placeholder*="email" i]').first
            await email_input.wait_for(state="visible", timeout=15000)
            await email_input.click()
            await email_input.fill(self.email)
            self.logger.info("Filled email")
            await page.wait_for_timeout(1000)

            # Step 2: Click Continue button (the black one below email, not the OAuth buttons)
            self.logger.info("Clicking Continue button...")
            # The Continue button is directly after the email input - it's a black button
            # Try multiple approaches to find the right Continue button
            continue_btn = None

            # Try to find the Continue button that's NOT an OAuth button
            all_continue_btns = page.locator('button:has-text("Continue")')
            count = await all_continue_btns.count()
            self.logger.info(f"Found {count} Continue buttons")

            for i in range(count):
                btn = all_continue_btns.nth(i)
                text = await btn.text_content()
                # Skip OAuth buttons (they have "Continue with X")
                if text and "with" not in text.lower():
                    continue_btn = btn
                    self.logger.info(f"Selected Continue button with text: {text}")
                    break

            if not continue_btn:
                # Fallback: press Enter in email field to submit
                self.logger.info("No standalone Continue button found, pressing Enter...")
                await email_input.press("Enter")
            else:
                await continue_btn.scroll_into_view_if_needed()
                await page.wait_for_timeout(500)
                await continue_btn.click(force=True)

            self.logger.info("Continue button clicked, waiting for password field...")

            # Step 3: Wait for password field to appear
            await page.wait_for_timeout(3000)

            # Take screenshot to debug current state
            await page.screenshot(path=str(self.download_dir / "patreon_step1.png"))

            # Find password input - try multiple selectors with longer waits
            password_selectors = [
                'input[type="password"]',
                'input[name="current-password"]',
                'input[placeholder*="password" i]',
                'input[placeholder*="Password"]',
                'input[autocomplete="current-password"]',
            ]

            password_input = None
            # Try for up to 15 seconds
            for attempt in range(5):
                self.logger.info(f"Looking for password field (attempt {attempt + 1}/5)...")
                for selector in password_selectors:
                    try:
                        pwd = page.locator(selector).first
                        if await pwd.is_visible(timeout=2000):
                            password_input = pwd
                            self.logger.info(f"Found password field with: {selector}")
                            break
                    except:
                        continue
                if password_input:
                    break
                await page.wait_for_timeout(1000)

            if not password_input:
                self.logger.error("Could not find password field after all attempts")
                await page.screenshot(path=str(self.download_dir / "patreon_no_password.png"))
                # Log page content for debugging
                content = await page.content()
                self.logger.info(f"Page contains 'password': {'password' in content.lower()}")
                return False

            # Step 4: Fill password
            await password_input.click()
            await password_input.fill(self.password)
            self.logger.info("Filled password")
            await page.wait_for_timeout(1500)

            # Dismiss any consent popups before clicking login
            await self._accept_cookies(page)

            # Step 5: Submit login form
            login_url = page.url
            self.logger.info("Submitting login form...")

            # Press Enter in password field (most reliable form submission)
            await password_input.press("Enter")
            self.logger.info("Pressed Enter to submit")

            # Step 6: Wait for response (URL change, code verification, or error)
            self.logger.info("Waiting for login response...")
            redirected = False
            for _ in range(15):
                await page.wait_for_timeout(1000)
                # Check for URL change (redirect to OAuth or fplreview)
                if page.url != login_url:
                    redirected = True
                    break
                # Check for email code verification page (Patreon 2FA from new device)
                try:
                    if await page.locator('text="Enter your login code"').is_visible(timeout=500):
                        return await self._handle_code_verification(page, login_url)
                except:
                    pass

            if not redirected:
                # Fallback: try clicking Continue/submit button
                self.logger.warning("No response after Enter, trying button click...")
                try:
                    submit_btn = page.locator('button[type="submit"], button:has-text("Continue")').first
                    await submit_btn.click(force=True)
                    for _ in range(10):
                        await page.wait_for_timeout(1000)
                        if page.url != login_url:
                            redirected = True
                            break
                        try:
                            if await page.locator('text="Enter your login code"').is_visible(timeout=500):
                                return await self._handle_code_verification(page, login_url)
                        except:
                            pass
                except:
                    pass

            if not redirected:
                self.logger.error("Login did not redirect - check credentials or CAPTCHA")
                await page.screenshot(path=str(self.download_dir / "patreon_login_error.png"))
                return False

            current_url = page.url
            self.logger.info(f"URL after login: {current_url[:120]}")

            if self._is_fplreview_url(current_url):
                self.logger.info("Successfully logged in and redirected to FPL Review")
                return True

            # Might need to authorize the app (OAuth Allow page)
            if self._is_patreon_url(current_url):
                try:
                    auth_btn = page.locator('button:has-text("Allow"), button:has-text("Authorize")').first
                    if await auth_btn.is_visible(timeout=5000):
                        await auth_btn.click()
                        self.logger.info("Clicked authorize button")
                        for _ in range(15):
                            await page.wait_for_timeout(1000)
                            if self._is_fplreview_url(page.url):
                                break
                except:
                    pass

            return self._is_fplreview_url(page.url)

        except Exception as e:
            self.logger.error(f"Error completing Patreon login: {e}")
            await page.screenshot(path=str(self.download_dir / "patreon_login_error.png"))
            return False

    async def _handle_code_verification(self, page: Page, login_url: str) -> bool:
        """Handle Patreon's email code verification (2FA triggered from new device/IP).

        When running headless, this cannot be completed automatically.
        The user must do a one-time manual login to establish a persistent session.
        """
        self.logger.info("Patreon requires email verification code")

        if self._headless:
            self.logger.error(
                "Cannot enter verification code in headless mode. "
                "Run a one-time manual login to save the session:\n"
                "  docker compose exec -it fpl-agent python -m src.fplreview_client"
            )
            await page.screenshot(path=str(self.download_dir / "patreon_login_error.png"))
            return False

        # Non-headless: wait for user to enter code in the browser window
        self.logger.info("Please enter the verification code sent to your email...")
        self.logger.info("Waiting up to 5 minutes for code entry...")

        for _ in range(300):  # 5 minutes
            await page.wait_for_timeout(1000)
            # Check if we moved past the code page
            try:
                if not await page.locator('text="Enter your login code"').is_visible(timeout=500):
                    break
            except:
                break
            if page.url != login_url:
                break

        await page.wait_for_timeout(3000)  # Wait for final redirect
        current_url = page.url
        self.logger.info(f"URL after code verification: {current_url[:120]}")

        if self._is_fplreview_url(current_url):
            self.logger.info("Successfully logged in after code verification")
            return True

        # Check for OAuth allow page
        if self._is_patreon_url(current_url):
            try:
                auth_btn = page.locator('button:has-text("Allow"), button:has-text("Authorize")').first
                if await auth_btn.is_visible(timeout=5000):
                    await auth_btn.click()
                    self.logger.info("Clicked authorize button")
                    for _ in range(15):
                        await page.wait_for_timeout(1000)
                        if self._is_fplreview_url(page.url):
                            return True
            except:
                pass

        return self._is_fplreview_url(page.url)

    async def _accept_cookies(self, page: Page) -> None:
        """Accept cookie consent popup if present (including Transcend consent manager)."""
        try:
            # First try to dismiss Transcend consent manager (Patreon uses this)
            transcend_selectors = [
                '#transcend-consent-manager button:has-text("Accept")',
                '#transcend-consent-manager button:has-text("Accept All")',
                '[data-testid="consent-manager-accept"]',
                'button[class*="accept"]',
                '#transcend-consent-manager [role="button"]',
            ]
            for selector in transcend_selectors:
                try:
                    btn = page.locator(selector).first
                    if await btn.is_visible(timeout=2000):
                        await btn.click(force=True)
                        self.logger.info(f"Dismissed Transcend consent via {selector}")
                        await page.wait_for_timeout(1000)
                        return
                except:
                    continue

            # Try to remove Transcend overlay via JavaScript if clicking fails
            try:
                removed = await page.evaluate("""() => {
                    const overlay = document.querySelector('#transcend-consent-manager');
                    if (overlay) {
                        overlay.remove();
                        return true;
                    }
                    return false;
                }""")
                if removed:
                    self.logger.info("Removed Transcend consent manager via JS")
                    await page.wait_for_timeout(500)
                    return
            except:
                pass

            # Standard cookie consent selectors
            cookie_selectors = [
                'button:has-text("Accept All")',
                'button:has-text("Accept")',
                'a:has-text("Accept All")',
                '[class*="cookie"] button',
            ]
            for selector in cookie_selectors:
                try:
                    btn = page.locator(selector).first
                    if await btn.is_visible(timeout=2000):
                        await btn.click()
                        self.logger.info("Accepted cookies")
                        await page.wait_for_timeout(1000)
                        return
                except:
                    continue
        except:
            pass

    async def _handle_fplreview_reconnect(self, page: Page) -> bool:
        """Handle FPL Review Patreon reconnection if required. Returns True if reconnect was performed."""
        try:
            self.logger.info("Checking if Patreon reconnect is needed...")
            await page.wait_for_timeout(2000)  # Wait for page to settle

            # Only reconnect if the page explicitly asks for it
            needs_reconnect = await page.evaluate("""() => {
                const text = (document.body.innerText || '').toLowerCase();
                return text.includes('reconnect') || text.includes('premium subscriber');
            }""")
            if not needs_reconnect:
                self.logger.info("No reconnect needed on this page")
                return False

            # Check if there's a "LOG IN WITH PATREON" button
            reconnect_selectors = [
                'a:has-text("LOG IN WITH PATREON")',
                'a:has-text("Log in with Patreon")',
                'a:has-text("log in with patreon")',
                'button:has-text("LOG IN WITH PATREON")',
            ]

            for selector in reconnect_selectors:
                try:
                    btn = page.locator(selector).first
                    if await btn.is_visible(timeout=3000):
                        self.logger.info("Found reconnect button, clicking...")
                        await btn.click()
                        await page.wait_for_timeout(3000)

                        # Handle Patreon page (could be OAuth allow OR login page)
                        if self._is_patreon_url(page.url):
                            self.logger.info(f"On Patreon page: {page.url[:120]}")
                            await page.screenshot(path=str(self.download_dir / "patreon_step2.png"))

                            if "/login" in page.url:
                                # Need full login (previous login didn't persist)
                                self.logger.info("On Patreon login page, completing full login...")
                                login_ok = await self._complete_patreon_login(page)
                                if not login_ok:
                                    self.logger.warning("Patreon login during reconnect failed")
                                    continue
                            else:
                                # OAuth authorization page - click Allow
                                allow_selectors = [
                                    'button:has-text("Allow")',
                                    'button:has-text("Authorize")',
                                    'button[data-tag="allow-button"]',
                                ]
                                for allow_sel in allow_selectors:
                                    try:
                                        allow_btn = page.locator(allow_sel).first
                                        if await allow_btn.is_visible(timeout=3000):
                                            self.logger.info(f"Found Allow button with: {allow_sel}")
                                            await allow_btn.click()
                                            self.logger.info("Clicked Allow button")
                                            await page.wait_for_timeout(5000)
                                            break
                                    except:
                                        continue

                            # Wait for redirect back to fplreview
                            self.logger.info("Waiting for redirect back to FPL Review...")
                            for _ in range(15):
                                if self._is_fplreview_url(page.url):
                                    break
                                await page.wait_for_timeout(1000)

                        self.logger.info(f"Current URL after auth: {page.url}")

                        # Accept cookies again if they appear
                        await self._accept_cookies(page)

                        # Check if we're now logged in (no reconnect button visible)
                        await page.screenshot(path=str(self.download_dir / "after_reconnect.png"))
                        self.logger.info("Reconnect completed")
                        return True
                except Exception as e:
                    self.logger.warning(f"Error with selector {selector}: {e}")
                    continue

            return False
        except Exception as e:
            self.logger.warning(f"Error handling reconnect: {e}")
            return False

    async def _navigate_to_planner(self, page: Page) -> None:
        """Navigate to the Free Planner via the Team Planner menu."""
        try:
            self.logger.info("Looking for Team Planner menu...")

            # First, hover over or click "Team Planner" to open dropdown
            team_planner_selectors = [
                'a:has-text("Team Planner")',
                'span:has-text("Team Planner")',
                'li:has-text("Team Planner")',
            ]

            for selector in team_planner_selectors:
                try:
                    menu_item = page.locator(selector).first
                    if await menu_item.is_visible(timeout=3000):
                        self.logger.info(f"Found Team Planner with: {selector}")
                        # Hover to open dropdown
                        await menu_item.hover()
                        await page.wait_for_timeout(1000)
                        break
                except:
                    continue

            # Now look for "Free Planner" in the dropdown
            self.logger.info("Looking for Free Planner option...")
            planner_selectors = [
                'a:has-text("Free Planner")',
                'a[href*="free-planner"]',
                'li:has-text("Free Planner") a',
            ]

            for selector in planner_selectors:
                try:
                    planner_link = page.locator(selector).first
                    if await planner_link.is_visible(timeout=3000):
                        self.logger.info(f"Found Free Planner with: {selector}")
                        await planner_link.click()
                        self.logger.info("Clicked Free Planner")
                        await page.wait_for_timeout(3000)
                        await page.wait_for_load_state("domcontentloaded", timeout=30000)
                        await self._trigger_data_load(page)
                        return
                except:
                    continue

            # Fallback: try direct navigation
            self.logger.info("Menu navigation failed, trying direct URL...")
            await page.goto(self.FREE_PLANNER_URL, wait_until="domcontentloaded", timeout=60000)
            await page.wait_for_timeout(2000)
            await self._trigger_data_load(page)

        except Exception as e:
            self.logger.warning(f"Error navigating to planner: {e}")
            # Fallback to direct URL
            await page.goto(self.FREE_PLANNER_URL, wait_until="domcontentloaded", timeout=60000)
            await self._trigger_data_load(page)

    async def _trigger_data_load(self, page: Page) -> None:
        """Trigger the player data table to load on the Free Planner page."""
        self.logger.info("Triggering data load on planner page...")

        # Fill the Team ID field before clicking Load Page
        team_id_selectors = [
            'input[id*="team" i]',
            'input[name*="team" i]',
            'input[placeholder*="team" i]',
            'input[placeholder*="Team ID" i]',
            'input[type="number"]',
            'input[type="text"]:near(:text("Team ID"))',
        ]
        for selector in team_id_selectors:
            try:
                input_el = page.locator(selector).first
                if await input_el.is_visible(timeout=2000):
                    await input_el.clear()
                    await input_el.fill(self.team_id)
                    self.logger.info(f"Filled Team ID field with {self.team_id} via '{selector}'")
                    break
            except:
                continue

        # Strategy 1: Click "Load Page" button (visible next to Team ID field)
        load_page_selectors = [
            'button:has-text("Load Page")',
            'input[value="Load Page"]',
            'a:has-text("Load Page")',
            'button:has-text("Load")',
        ]
        for selector in load_page_selectors:
            try:
                btn = page.locator(selector).first
                if await btn.is_visible(timeout=3000):
                    self.logger.info(f"Found Load Page button with: {selector}")
                    await btn.click()
                    self.logger.info("Clicked Load Page button")
                    await page.wait_for_timeout(3000)
                    if await self._wait_for_table_data(page):
                        return
                    break
            except:
                continue

        # Strategy 2: Interact with "Load Group" dropdown to force data refresh
        load_group_selectors = [
            'select:near(:text("Load Group"))',
            'select[id*="group" i]',
            'select[name*="group" i]',
            '#load_group',
        ]
        for selector in load_group_selectors:
            try:
                dropdown = page.locator(selector).first
                if await dropdown.is_visible(timeout=2000):
                    self.logger.info(f"Found Load Group dropdown with: {selector}")
                    # Select the first option to trigger a data load
                    options = await dropdown.locator("option").all_text_contents()
                    if options:
                        await dropdown.select_option(index=0)
                        self.logger.info(f"Selected Load Group option: {options[0] if options else 'first'}")
                        await page.wait_for_timeout(3000)
                        if await self._wait_for_table_data(page):
                            return
                    break
            except:
                continue

        # Strategy 3: Click the "EV" display mode button
        ev_selectors = [
            'button:has-text("EV")',
            'a:has-text("EV")',
            'input[value="EV"]',
            ':text-is("EV")',
        ]
        for selector in ev_selectors:
            try:
                btn = page.locator(selector).first
                if await btn.is_visible(timeout=2000):
                    self.logger.info(f"Found EV button with: {selector}")
                    await btn.click()
                    self.logger.info("Clicked EV button")
                    await page.wait_for_timeout(3000)
                    if await self._wait_for_table_data(page):
                        return
                    break
            except:
                continue

        # Strategy 4: JavaScript - dispatch change events on select elements to trigger data refresh
        self.logger.info("Trying JS-based data trigger...")
        await page.evaluate("""() => {
            // Dispatch change events on all select elements
            document.querySelectorAll('select').forEach(sel => {
                sel.dispatchEvent(new Event('change', { bubbles: true }));
            });
            // Also try triggering any load buttons via click
            document.querySelectorAll('button, input[type="button"], input[type="submit"]').forEach(btn => {
                const text = (btn.textContent || btn.value || '').toLowerCase();
                if (text.includes('load') || text.includes('refresh') || text.includes('update')) {
                    btn.click();
                }
            });
        }""")
        await page.wait_for_timeout(5000)
        await self._wait_for_table_data(page)

    async def _wait_for_table_data(self, page: Page, timeout_s: int = 30, min_rows: int = 5) -> bool:
        """Poll for table rows to appear in the DOM. Returns True if data found."""
        self.logger.info(f"Waiting for table data to appear (min {min_rows} rows)...")

        table_row_selectors = [
            'table tbody tr',
            'table tr:not(:first-child)',
            '.player-row',
            '[class*="player"] tr',
            'table tr td',
        ]

        for attempt in range(timeout_s // 2):
            # Check via CSS selectors
            for selector in table_row_selectors:
                try:
                    count = await page.locator(selector).count()
                    if count >= min_rows:
                        self.logger.info(f"Table data found: {count} rows via '{selector}'")
                        return True
                except:
                    continue

            # Check via page.evaluate for any table with data rows
            has_data = await page.evaluate("""() => {
                const tables = document.querySelectorAll('table');
                for (const table of tables) {
                    const rows = table.querySelectorAll('tbody tr, tr');
                    // Filter out header-only tables
                    if (rows.length > 2) {
                        // Check that rows have actual cell content (not empty)
                        for (const row of rows) {
                            const cells = row.querySelectorAll('td');
                            if (cells.length > 3) {
                                const text = Array.from(cells).map(c => c.textContent.trim()).join('');
                                if (text.length > 0) return true;
                            }
                        }
                    }
                }
                return false;
            }""")
            if has_data:
                self.logger.info("Table data found via JS evaluation")
                return True

            await page.wait_for_timeout(2000)

        self.logger.warning(f"No table data found after {timeout_s}s")
        return False

    async def _download_csv(self, page: Page, context) -> Optional[Path]:
        """Find and click the CSV download button."""
        try:
            self.logger.info("Waiting for page to fully load...")
            await page.wait_for_load_state("domcontentloaded", timeout=30000)
            await page.wait_for_timeout(5000)  # Wait for dynamic content

            # Take screenshot to see current state
            await page.screenshot(path=str(self.download_dir / "planner_page.png"), full_page=True)

            # Scroll down to find the download button
            self.logger.info("Scrolling down to find download button...")
            for _ in range(15):
                await page.evaluate("window.scrollBy(0, 400)")
                await page.wait_for_timeout(300)

            # Try to expand/load the player list if there's a button or tab for it
            try:
                # Look for tabs or buttons that might show full player data
                player_list_selectors = [
                    'text="Player List"',
                    'text="All Players"',
                    'button:has-text("Player")',
                    'a:has-text("Player List")',
                    '[class*="tab"]:has-text("Player")',
                    'text="Show All"',
                ]
                for selector in player_list_selectors:
                    try:
                        btn = page.locator(selector).first
                        if await btn.is_visible(timeout=2000):
                            self.logger.info(f"Clicking to expand player data: {selector}")
                            await btn.click()
                            await page.wait_for_timeout(3000)
                            break
                    except:
                        continue
            except:
                pass

            # Scroll through page to trigger lazy loading
            self.logger.info("Scrolling to trigger lazy loading...")
            for _ in range(5):
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await page.wait_for_timeout(1000)

            # Look specifically for "Download Data (CSV)" button - be very specific
            self.logger.info("Looking for CSV download button...")
            download_btn = None

            # First, try to find by exact text match using JavaScript
            try:
                download_link = await page.evaluate("""() => {
                    const links = document.querySelectorAll('a');
                    for (const link of links) {
                        const text = link.textContent.trim();
                        if (text === 'Download Data (CSV)' || text === 'Download CSV') {
                            return true;
                        }
                    }
                    return false;
                }""")
                if download_link:
                    # Use text-is for exact match
                    download_btn = page.locator('a:text-is("Download Data (CSV)")').first
                    if not await download_btn.is_visible(timeout=2000):
                        download_btn = page.locator('a:text-is("Download CSV")').first
                    self.logger.info("Found download button via exact text match")
            except:
                pass

            # Fallback to pattern matching, but filter out Upload buttons
            if not download_btn:
                download_selectors = [
                    'a:has-text("Download Data")',
                    'button:has-text("Download Data")',
                    'a:has-text("Download CSV")',
                    'button:has-text("Download CSV")',
                    '[download*=".csv"]',
                    'a[href*=".csv"]',
                ]

                for selector in download_selectors:
                    try:
                        elements = page.locator(selector)
                        count = await elements.count()
                        if count > 0:
                            for i in range(count):
                                btn = elements.nth(i)
                                if await btn.is_visible(timeout=1000):
                                    text = (await btn.text_content() or "").strip()
                                    # Skip if it contains "Upload"
                                    if "upload" in text.lower():
                                        continue
                                    self.logger.info(f"Found element with selector '{selector}': {text[:50]}")
                                    download_btn = btn
                                    break
                        if download_btn:
                            break
                    except:
                        continue

            if not download_btn:
                # No download button - try DOM scrape as last resort
                self.logger.warning("Could not find CSV download button, trying DOM scrape...")
                await page.screenshot(path=str(self.download_dir / "no_download_btn.png"), full_page=True)
                links = await page.locator("a").all_text_contents()
                self.logger.info(f"Links on page: {links[:20]}")
                return await self._scrape_table_to_csv(page)

            # Scroll the button into view and wait
            await download_btn.scroll_into_view_if_needed()
            await page.wait_for_timeout(1000)

            # Set up download handling
            csv_path = self.download_dir / "projected_points.csv"

            # Track existing *.csv files before download (broad glob)
            system_downloads = Path.home() / "Downloads"
            existing_files = set(self.download_dir.glob("*.csv"))
            existing_system_files = set(system_downloads.glob("*.csv")) if system_downloads.exists() else set()

            # Inject JS interceptor to capture blob/data URL downloads
            await page.evaluate("""() => {
                window.__capturedCSV = null;
                // Intercept Blob constructor to capture CSV data at creation
                const OrigBlob = window.Blob;
                window.Blob = function(parts, options) {
                    const blob = new OrigBlob(parts, options);
                    const type = (options && options.type) || '';
                    if (type.includes('csv') || type.includes('text/plain') || type.includes('octet')) {
                        try {
                            const content = parts.map(p => typeof p === 'string' ? p : '').join('');
                            if (content.length > 50 && content.includes(',') && content.includes('\\n')) {
                                window.__capturedCSV = content;
                            }
                        } catch(e) {}
                    }
                    return blob;
                };
                window.Blob.prototype = OrigBlob.prototype;
                // Intercept createElement('a').click() blob downloads
                const origCreateElement = document.createElement.bind(document);
                document.createElement = function(tag) {
                    const el = origCreateElement(tag);
                    if (tag.toLowerCase() === 'a') {
                        const origClick = el.click.bind(el);
                        el.click = function() {
                            const href = el.href || '';
                            if (href.startsWith('blob:') || href.startsWith('data:')) {
                                // Capture blob URL content
                                if (href.startsWith('blob:')) {
                                    fetch(href).then(r => r.text()).then(t => {
                                        window.__capturedCSV = t;
                                    }).catch(() => {});
                                } else if (href.startsWith('data:')) {
                                    // data:text/csv;charset=utf-8,...
                                    const commaIdx = href.indexOf(',');
                                    if (commaIdx > -1) {
                                        try {
                                            window.__capturedCSV = decodeURIComponent(href.substring(commaIdx + 1));
                                        } catch(e) {
                                            window.__capturedCSV = href.substring(commaIdx + 1);
                                        }
                                    }
                                }
                            }
                            return origClick();
                        };
                    }
                    return el;
                };
                // Also intercept window.URL.createObjectURL
                const origCreateObjectURL = URL.createObjectURL.bind(URL);
                URL.createObjectURL = function(blob) {
                    const url = origCreateObjectURL(blob);
                    if (blob instanceof Blob) {
                        blob.text().then(t => {
                            if (t.includes(',') && (t.includes('\\n') || t.includes('Pos') || t.includes('Name'))) {
                                window.__capturedCSV = t;
                            }
                        }).catch(() => {});
                    }
                    return url;
                };
            }""")
            self.logger.info("Injected JS download interceptor")

            # Handle dialog with a delay
            async def handle_dialog(dialog):
                self.logger.info(f"Dialog appeared: {dialog.message[:100] if dialog.message else 'No message'}...")
                await page.wait_for_timeout(500)
                await dialog.accept()
                self.logger.info("Dialog accepted")

            page.on("dialog", lambda d: asyncio.create_task(handle_dialog(d)))

            # Use expect_download() context manager for reliable download handling
            self.logger.info("Clicking download button and waiting for download...")
            try:
                async with page.expect_download(timeout=5000) as download_info:
                    await download_btn.click()

                download = await download_info.value
                self.logger.info(f"Download started: {download.suggested_filename}")

                await download.save_as(str(csv_path))
                self.logger.info(f"CSV downloaded successfully to {csv_path}")
                return csv_path

            except Exception as e:
                self.logger.warning(f"expect_download failed: {e}, trying fallback methods...")

                # Fallback tier 1: Check JS interceptor for captured CSV content
                await page.wait_for_timeout(2000)
                captured_csv = await page.evaluate("() => window.__capturedCSV")
                if captured_csv and len(captured_csv) > 50:
                    self.logger.info(f"Captured CSV via JS interceptor ({len(captured_csv)} chars)")
                    csv_path.write_text(captured_csv, encoding="utf-8")
                    self.logger.info(f"Saved intercepted CSV to {csv_path}")
                    return csv_path

                # Fallback tier 2: Click again and scan for ANY new .csv file
                await download_btn.click()
                self.logger.info("Clicked download button again, scanning for new CSV files...")

                for i in range(10):  # Wait up to 10 seconds
                    await page.wait_for_timeout(1000)

                    # Check JS interceptor again
                    captured_csv = await page.evaluate("() => window.__capturedCSV")
                    if captured_csv and len(captured_csv) > 50:
                        self.logger.info(f"Captured CSV via JS interceptor ({len(captured_csv)} chars)")
                        csv_path.write_text(captured_csv, encoding="utf-8")
                        return csv_path

                    # Check data directory for any new CSV
                    current_files = set(self.download_dir.glob("*.csv"))
                    new_files = current_files - existing_files
                    if new_files:
                        new_file = list(new_files)[0]
                        self.logger.info(f"Found downloaded file in data dir: {new_file}")
                        if new_file != csv_path:
                            import shutil
                            shutil.move(str(new_file), str(csv_path))
                            self.logger.info(f"Moved to {csv_path}")
                        return csv_path

                    # Check system Downloads folder for any new CSV
                    if system_downloads.exists():
                        current_system_files = set(system_downloads.glob("*.csv"))
                        new_system_files = current_system_files - existing_system_files
                        if new_system_files:
                            new_file = list(new_system_files)[0]
                            self.logger.info(f"Found downloaded file in Downloads: {new_file}")
                            import shutil
                            shutil.move(str(new_file), str(csv_path))
                            self.logger.info(f"Moved to {csv_path}")
                            return csv_path

                # Fallback tier 3: Scrape the DOM table directly
                self.logger.warning("All download methods failed, trying DOM table scrape...")
                scraped = await self._scrape_table_to_csv(page)
                if scraped:
                    return scraped

                self.logger.error("All download and scrape methods failed")
                await page.screenshot(path=str(self.download_dir / "download_failed.png"), full_page=True)
                return None

        except Exception as e:
            self.logger.error(f"Error downloading CSV: {e}")
            await page.screenshot(path=str(self.download_dir / "download_error.png"), full_page=True)
            return None

    async def _scrape_table_to_csv(self, page: Page) -> Optional[Path]:
        """Extract table data directly from the DOM and write to CSV. Last-resort fallback."""
        try:
            self.logger.info("Attempting to scrape table data from DOM...")

            table_data = await page.evaluate("""() => {
                // Find the largest table on the page (most likely the player data table)
                const tables = document.querySelectorAll('table');
                let bestTable = null;
                let bestRowCount = 0;

                for (const table of tables) {
                    const rows = table.querySelectorAll('tr');
                    if (rows.length > bestRowCount) {
                        bestRowCount = rows.length;
                        bestTable = table;
                    }
                }

                if (!bestTable || bestRowCount < 3) return null;

                const result = [];
                const rows = bestTable.querySelectorAll('tr');
                for (const row of rows) {
                    const cells = row.querySelectorAll('th, td');
                    const rowData = [];
                    for (const cell of cells) {
                        // Get text content, trim whitespace
                        let text = cell.textContent.trim();
                        // Escape commas and quotes for CSV
                        if (text.includes(',') || text.includes('"') || text.includes('\\n')) {
                            text = '"' + text.replace(/"/g, '""') + '"';
                        }
                        rowData.push(text);
                    }
                    if (rowData.length > 0) {
                        result.push(rowData.join(','));
                    }
                }
                return result.length > 2 ? result.join('\\n') : null;
            }""")

            if not table_data:
                self.logger.warning("No table data found in DOM to scrape")
                return None

            csv_path = self.download_dir / "projected_points.csv"
            csv_path.write_text(table_data, encoding="utf-8")
            self.logger.info(f"Scraped table data saved to {csv_path} ({len(table_data)} chars)")
            return csv_path

        except Exception as e:
            self.logger.error(f"Error scraping table: {e}")
            return None


async def download_fplreview_projections(
    email: str,
    password: str,
    download_dir: Path,
    headless: bool = True,
    logger: Optional[logging.Logger] = None,
    team_id: str = "",
) -> Optional[Path]:
    """
    Convenience function to download FPL Review projections.

    Args:
        email: Patreon email
        password: Patreon password
        download_dir: Directory to save the CSV
        headless: Run browser in headless mode
        logger: Optional logger instance
        team_id: FPL team ID to load data for

    Returns:
        Path to downloaded CSV, or None if failed
    """
    client = FPLReviewClient(email, password, download_dir, logger, team_id=team_id)
    return await client.download_projections_csv(headless=headless)


if __name__ == "__main__":
    # Test the client directly
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    sys.path.insert(0, str(Path(__file__).parent))

    from src.config import Config

    async def main():
        config = Config.from_env()

        if not config.fpl_review_email or not config.fpl_review_password:
            print("Error: FPL_REVIEW_EMAIL and FPL_REVIEW_PASSWORD must be set in .env")
            return

        print(f"Downloading projections with email: {config.fpl_review_email}")

        csv_path = await download_fplreview_projections(
            email=config.fpl_review_email,
            password=config.fpl_review_password,
            download_dir=config.data_dir,
            headless=False,  # Set to False for debugging
            logger=config.logger,
            team_id=str(config.fpl_team_id),
        )

        if csv_path:
            print(f"Success! CSV saved to: {csv_path}")
        else:
            print("Failed to download CSV")

    asyncio.run(main())
